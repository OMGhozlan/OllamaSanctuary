import os
import subprocess
import pymupdf
import time
import logging
from pathlib import Path
import typer
import pyfiglet
from yaspin import yaspin
from inquirer import list_input
from difflib import get_close_matches

from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage

app = typer.Typer()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

SYSTEM_PROMPT = ""

def translate(llm, text, src_language, tgt_language):
    """
    Translate the given text from the source language to the target language using the
    provided language model.

    Args:
        llm: The language model to use for translation.
        text: The text to translate.
        src_language: The source language of the text.
        tgt_language: The target language to translate the text to.

    Returns:
        The translated text.
    """
    messages = [
        (
         "system",
         SYSTEM_PROMPT,
        ),
        ("human", f'Translate this from {src_language} to {tgt_language}: {text}'),
    ]
    return llm.invoke(messages).content



def draw_target_text(page, block, ocg_xref, target):
    """
    Draw target translation in place of given block.

    Args:
        page: The page object to draw on.
        block (tuple): The text block with its bounding box.
        ocg_xref: The optional content group reference.
    """
    bbox = block[:4]
    page.draw_rect(bbox, color=None, fill=pymupdf.pdfcolor["white"], oc=ocg_xref)
    page.insert_htmlbox(
        bbox, target, css="* {font-family: sans-serif;}", oc=ocg_xref
    )

def check_model_if_exists(model_name):
    """Check if the specified model exists.

    Args:
        model_name: Name of the model to check.

    Returns:
        None. Raises an error if the model does not exist.
    """
    import ollama
    model_names = [model['model'] for model in ollama.list()['models']]

    if model_name not in model_names:
        suggestions = get_close_matches(model_name, model_names, n=3)
        suggestion_message = (f"Model '{model_name}' not found. Did you mean: "
                              f"{', '.join(suggestions)}?" if suggestions else "No similar models found.")
        logging.error(suggestion_message)
        raise RuntimeError(suggestion_message)

def process_pdf(file_path, model_name, src_language, tgt_language, n=10):
    """Process a given PDF and translate it to a given language, splitting the PDF into smaller parts first.

    Args:
        file_path: Path to the PDF file.
        model_name: Name of the model to use.
        src_language: Source language of the text.
        tgt_language: Target language for translation.
        n: Number of pages per smaller PDF file. If None, process the entire PDF as a single document.
    """
    logging.info(f"Processing {file_path}")
    
    # Open the original document
    doc = pymupdf.open(file_path)
    total_pages = len(doc)

    logging.info(f"Found {total_pages} pages in {file_path}")

    # Folder to store the smaller PDFs
    output_folder = Path("Original")
    output_folder.mkdir(parents=True, exist_ok=True)

    # Split the document into smaller parts and save them
    sub_pdf_paths = []
    translated_sub_pdf_paths = [] 
    
    for start_page in range(0, total_pages, n):
        sub_pdf_path = output_folder / f"{file_path.stem}_part{start_page // n + 1}.pdf"
        sub_doc = pymupdf.open()  # New document for each part
        sub_doc.insert_pdf(doc, from_page=start_page, to_page=min(start_page + n - 1, total_pages - 1))
        sub_doc.save(sub_pdf_path)
        sub_pdf_paths.append(sub_pdf_path)
        logging.info(f"Saved sub-PDF: {sub_pdf_path}")

    # Now process each sub-PDF
    start_time = time.time()
    llm = ChatOllama(model=model_name, temperature=0.5, num_predict=500)

    with yaspin(text="Processing pages", color="cyan") as spinner:
        for sub_pdf_path in sub_pdf_paths:
            sub_doc = pymupdf.open(sub_pdf_path)
            sub_total_pages = len(sub_doc)
            logging.info(f"Processing sub-PDF: {sub_pdf_path} with {sub_total_pages} pages")
            
            # Process each page in the sub-PDF
            for idx, page in enumerate(sub_doc):
                spinner.text = f"Processing page {idx + 1}/{sub_total_pages} of {sub_pdf_path}"
                blocks = page.get_text("blocks", flags=pymupdf.TEXT_DEHYPHENATE)
                ocg_xref = sub_doc.add_ocg(f"{tgt_language.capitalize()}", on=True)
                
                for block in blocks:
                    try:
                        text = block[4]
                        target = translate(llm, text, src_language, tgt_language)
                        if "Error" in target:
                            logging.error(target)
                        else:
                            draw_target_text(page, block, ocg_xref, target)
                    except Exception as e:
                        logging.error(f"Error processing page {idx + 1}: {e}")

                elapsed_time = time.time() - start_time
                avg_time_per_page = elapsed_time / (idx + 1)
                estimated_total_time = avg_time_per_page * sub_total_pages
                estimated_time_left = estimated_total_time - elapsed_time

                spinner.text = (f"{100 * (idx + 1) / sub_total_pages:.2f}% complete - "
                                f"Elapsed: {elapsed_time:.2f}s, "
                                f"ETA: {estimated_time_left:.2f}s")

            # After processing each sub-PDF, save the translated version
            translated_sub_pdf_path = sub_pdf_path.with_name(f"{sub_pdf_path.stem}_translated.pdf")
            sub_doc.save(translated_sub_pdf_path)
            translated_sub_pdf_paths.append(translated_sub_pdf_path)
            logging.info(f"Saved translated sub-PDF: {translated_sub_pdf_path}")
            
    # Merge all translated sub-PDFs into one final PDF
    merged_pdf = pymupdf.open()  # Start with a new blank PDF document
    for translated_sub_pdf_path in translated_sub_pdf_paths:
        translated_sub_doc = pymupdf.open(translated_sub_pdf_path)
        merged_pdf.insert_pdf(translated_sub_doc)  # Insert pages from each translated sub-PDF

    # Define the merged output path
    merged_output_path = Path(f"{file_path.stem}_translated_merged.pdf")
    merged_pdf.save(merged_output_path)

    # After merging, delete the translated sub-PDFs and original sub-PDFs
    for sub_pdf_path in sub_pdf_paths:
        logging.info(f"Deleting original sub-PDF: {sub_pdf_path}")
        os.remove(sub_pdf_path)
    
    for translated_sub_pdf_path in translated_sub_pdf_paths:
        logging.info(f"Deleting translated sub-PDF: {translated_sub_pdf_path}")
        os.remove(translated_sub_pdf_path)

    # After merging, delete the "Original" folder if empty
    if not os.listdir(output_folder):
        os.rmdir(output_folder)

    logging.info(f"Merged translated PDF saved to: {merged_output_path}")
    logging.info("Processing completed!")
    
    doc.subset_fonts()
    return doc


def is_ollama_running():
    try:
        # Check if Ollama is running by listing the processes
        result = subprocess.run(['tasklist'], capture_output=True, text=True)
        return 'ollama.exe' in result.stdout
    except Exception as e:
        typer.echo(f"Error checking Ollama status: {e}")
        return False

def start_ollama():
    try:
        subprocess.Popen(['ollama', 'serve'])
        typer.echo("Ollama started successfully.")
    except Exception as e:
        typer.echo(f"Error starting Ollama: {e}")
        raise typer.Exit()

@app.command()
def main(model_name="qwen2.5:7b", extension=".pdf"):
    global SYSTEM_PROMPT
    """Main entry point for the script."""
    print(pyfiglet.figlet_format("PDF Translate"))

    if not is_ollama_running():
        start_ollama()

    file_path = Path(typer.prompt("Enter the path to the PDF file"))
    if not file_path.is_file() or not file_path.suffix == ".pdf":
        typer.echo("Invalid PDF file. Please try again.")
        raise typer.Exit()

    prompt_files = [f for f in os.listdir('.') if os.path.isfile(f) and f.endswith(('.txt', '.prompt'))]

    if prompt_files:
        prompt_file = Path(list_input("Select the prompt file", choices=prompt_files))

        if not prompt_file.is_file():
            typer.echo("Invalid prompt file. Please try again.")
            raise typer.Exit()

        try:
            with open(prompt_file, 'r') as file:
                SYSTEM_PROMPT = file.read()
        except Exception as e:
            typer.echo(f"Error reading prompt file: {e}")
            raise typer.Exit()

    else:
        # Default to 'System.prompt' if no prompt files are found
        system_prompt_path = Path(__file__).parent / "System.prompt"

        if system_prompt_path.is_file():
            try:
                with open(system_prompt_path, 'r') as file:
                    SYSTEM_PROMPT = file.read()
            except Exception as e:
                typer.echo(f"Error reading System.prompt file: {e}")
                raise typer.Exit()
        else:
            # Fallback: Prompt user to enter the prompt manually
            SYSTEM_PROMPT = typer.prompt("System.prompt file not found. Enter your prompt directly:")

    src_language = typer.prompt("Enter the source language code (e.g. English)")
    tgt_language = typer.prompt("Enter the target language code (e.g. Deutsch)")

    n = typer.prompt("Enter the number of pages per smaller PDF file (leave blank for entire document)", default=None)
    n = int(n) if n else None

    check_model_if_exists(model_name)
    doc = process_pdf(file_path, model_name, src_language, tgt_language, n)
    output_file_path = f'{file_path.stem}_{src_language}-{tgt_language}'
    output_file_path = file_path.with_name(f"{output_file_path}{file_path.suffix}")
    doc.ez_save(output_file_path)
    typer.echo(f"File saved to {output_file_path}")

if __name__ == "__main__":
    app()
