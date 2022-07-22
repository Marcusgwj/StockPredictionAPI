<div id="top"></div>

<!-- PROJECT LOGO -->
<br />

<h3 align="center">Stock prediction models API</h3>

## About The Project

Stock predictions are made using the following models:

- Long Short-Term Memory
- Linear Regression
- Support Vector Regression

<p align="right">(<a href="#top">back to top</a>)</p>

### Built With

- [Python](https://www.python.org/)
- [scikit-learn](https://scikit-learn.org)
- [Keras](https://keras.io/)
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Flask](https://flask.palletsprojects.com/en/2.1.x/)

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->

## Usage

Make a Post request to each of the following routes:

- Long Short-Term Memory : https://stock-predictionapi.herokuapp.com/lstm
- Linear Regression: https://stock-predictionapi.herokuapp.com/lr
- Support Vector Regression: https://stock-predictionapi.herokuapp.com/svr

Include the ticker symbol of the stock in the request body.

For example:

{
"ticker" : "AAPL"
}

<p align="right">(<a href="#top">back to top</a>)</p>
