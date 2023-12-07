from flask import Flask, render_template
import pandas as pd
import numpy as np
import google.generativeai as palm
import yfinance as yf
import requests
from langchain.embeddings.google_palm import GooglePalmEmbeddings
from langchain.llms import GooglePalm
from langchain.chains.question_answering import load_qa_chain

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")
