<div id="top"></div>

<!-- PROJECT LOGO -->
<br />

<h3 align="center">Stock prediction models API</h3>

## About The Project

Stock predictions are made using the following models:

- Long short-term memory
- Linear regression
- Support vector machine

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

Make a post request to each of the following routes:

- Long short-term memory : https://stock-predictionapi.herokuapp.com/lstm
- Linear regression: https://stock-predictionapi.herokuapp.com/lr
- Support vector machine: https://stock-predictionapi.herokuapp.com/svm

Include the ticker symbol of the stock in the request body.

For example:

{
"ticker" : "AAPL"
}

<p align="right">(<a href="#top">back to top</a>)</p>
