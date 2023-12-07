from flask import Flask, render_template
import pandas as pd
import numpy as np
import google.generativeai as palm
import yfinance as yf
import requests
from langchain.embeddings.google_palm import GooglePalmEmbeddings
from langchain.llms import GooglePalm
from langchain.chains.question_answering import load_qa_chain
import datetime as dt
import statsmodels.api as sm
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
import math
import keras
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import pandas_datareader.data as web

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")
