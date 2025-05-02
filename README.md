
# The Verification Dashboard

The Verification Dashboard is designed to streamline multiple verification steps in one convenient location. It integrates features such as reverse image search, OCR, landmark detection, and automated geolocation. Leveraging the power of the Google Cloud Vision API and the OpenAI API, the dashboard enables fast, efficient access to valuable information, all in a single interface.


## 🤖 About Google Cloud Vision API

Google Cloud Vision API is a machine learning-powered image analysis service that provides developers with tools to understand the contents of an image. It can detect objects, faces, text, logos, and more within an image.



## 🚀 Getting Started

Before using the app, you need to obtain a Google Cloud Vision API key.

- Go to the [Google Cloud Platform Console](https://console.cloud.google.com/).
- Create a new project or select an existing one.
- Enable the Cloud Vision API for your project.
- Create a service account and download a private key in JSON format.
- Upload your Google Cloud Vision API key in JSON format by clicking on the `Upload a config file` button in the sidebar.


## 🛠️ Installation

To install the dependencies, simply run the following command:

`pip install -r requirements.txt`


## 🏃‍♀ Running the app

You can run the app locally by running the following command:

`streamlit run gvision.py`


## 🔎 Usage

Using GVision is simple and straightforward:

- Upload your Google Cloud Vision API key in JSON format by clicking on the `Upload a config file` button in the sidebar.
- Once the key is uploaded, the app will automatically authenticate with the Google Cloud Vision API.
- Upload an image in JPG, JPEG, or PNG format by clicking on the `Choose an image` button.
- Wait for the app to analyze the image. The app will detect landmarks and web entities present in the image and display them on a map.
- Choose between the different tile options to view the detected landmarks and web entities.

You can also find links to the Google Cloud Vision API documentation and pricing in the `Resources` section of the sidebar. 

To reset the app to its default state or to clear the uploaded image and results, click on the `Reset app` button.


## 📚 Resources

- [Google Cloud Vision API Documentation](https://cloud.google.com/vision/docs)
- [Google Cloud Vision API Pricing](https://cloud.google.com/vision/pricing)
