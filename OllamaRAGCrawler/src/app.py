
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from concurrent.futures import ThreadPoolExecutor, as_completed
from selenium.common.exceptions import NoSuchElementException
from prettylogger import log


import bs4
import requests
from langchain_community.document_loaders import WebBaseLoader


strainer = bs4.SoupStrainer('p')
loader = WebBaseLoader(
    web_paths=('https://en.wikipedia.org/wiki/Python_(programming_language)',),
    bs_kwargs={'parse_only': strainer}
)
log.info('Loading document')
docs = loader.load()
log.info(f'Document loaded: {docs}')