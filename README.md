![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# README

## NLP Project: Customer Reviews Classification, Product Category Clustering, and Summary Generation

## Introduction

This project aims to classify customer reviews into positive, neutral, or negative categories and provide product recommendations through summarization. The goal is to analyze and process textual data from customer reviews to build a system that can classify product reviews, cluster similar products, and generate helpful summaries.

We leverage advanced techniques such as Transformer models (like DistilBERT and RoBERTa**) for classification and fine-tuning, along with **OpenAI GPT-3.5 for generating textual summaries. The project covers all stages from data preprocessing, model training, evaluation, to product recommendation generation.

## Technologies Used

- Python: Main programming language for data processing and model implementation.
- pandas: For data manipulation and cleaning.
- scikit-learn: For machine learning models.
- Transformers: For working with pre-trained Transformer models like DistilBERT and RoBERTa.
- HuggingFace Transformers: For leveraging pre-trained Transformer models like DistilBERT and RoBERTa.
- OpenAI GPT-3.5: For generating product recommendations and summaries based on customer reviews.
- Matplotlib and Seaborn: For data visualization.
- Google Colab (Optional): For running the project in a cloud environment.

You can install the required packages using the following command:

pip install pandas seaborn matplotlib transformers scikit-learn openai

## Project Structure

This project consists of the following key components:

1. Data Preprocessing: This involves cleaning the review data, handling missing values, and transforming the data into a format suitable for model training.
   
2. Text Classification: Using pre-trained Transformer models, we classify customer reviews as positive, negative, or neutral.
   
3. Product Clustering: Products are grouped into categories based on their attributes using clustering techniques.
   
4. Review Summarization: Generating blog-style summaries of the top-rated products in each category using GPT-3.5, which also highlights differences, complaints, and offers recommendations.
   
5. Visualization: Various plots are generated to understand the distribution of review ratings and to visualize other aspects of the data.

## How to Run the Code

### 1. Clone the Repository  

Start by cloning the repository to your local machine or environment:
git clone <repository-url>
cd <repository-folder>

### 2. Install Required Libraries  

Install all necessary libraries using pip:
pip install -r requirements.txt

### 3. Set Up the API Key for OpenAI 

Make sure to set up your OpenAI API key in the environment. You can do this by adding it to the script or configuring it as an environment variable.
export OPENAI_API_KEY="your_api_key"

Alternatively, you can pass the API key directly in the script, as shown in the provided code.

### 4. Run the Code  

After setting up your environment, you can run the code to classify reviews, cluster products, and generate summaries. Execute the following:
python main.py

### 5. Results  

The results will be printed in the console, showing the blog-style summaries for each product category, as well as visualizations of the ratings distribution.

## Important Notes

- Data: Ensure that you have the dataset available in the working directory, as the code relies on this file for training and generating results.
  
- Model Fine-Tuning: Fine-tuning of the Transformer models (DistilBERT and RoBERTa) is performed on the dataset of reviews. You may need a GPU for faster processing, depending on the size of the data and models.

- Customization: You can modify the code to suit your specific needs, such as adjusting the number of products to display in each category or fine-tuning the models further with additional data.

## Conclusion

This project was designed to provide insights into how to process customer review data using classification models, product clustering techniques, and text summarization models. We covered every step of the pipeline from data cleaning, model training, evaluation, to generating product summaries using generative models like GPT-3.5.

If you'd like to modify or improve the project, feel free to interact with the data and models according to your specific use case.

🔗(  https://project-3-nlp-business-case-automated-customers-reviews-as3gwa.streamlit.app/ ) This is my project for analyzing and classifying product reviews using AI. Feel free to try it out and share it with your friends!
  
