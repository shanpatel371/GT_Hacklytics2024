# Hotel Search RAG Model 

## Description
Built a hotel search model that continuously collects data from the dataset and trains itself with it. The dataset is supposed to constantly do web scraping for new information on hotels, but due to limited storage capacity, that feature wasn't added. 

## Features
- Built a Retrieval Augmented Generation-based system that takes semantic queries of hotel criteria as inputs.
- Used **pandas** to manipulate and access a designated hotel dataset and **Jupyter Notebook** to visualize the results for testing.
- Utilized the **chromadb** embedding model to retrieve searches from the database and a decoder model to explain preferences to users.
