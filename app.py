from flask import Flask, render_template
import pandas as pd
import numpy as np
import google.generativeai as palm
import yfinance as yf
import requests
import chromadb
from chromadb.api.types import Documents, Embeddings

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")
