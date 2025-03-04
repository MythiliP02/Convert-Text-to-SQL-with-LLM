import os
import re
import sqlite3
import pandas as pd
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import HumanMessage, SystemMessage
import nltk
from nltk.corpus import stopwords, wordnet
 
# Initialize NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
 
# Initialize environment variables and the model
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["NVIDIA_API_KEY"] = "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"
 
model = ChatNVIDIA(model="meta/llama3-70b-instruct")
 
# Database connection
database = "northwind.db"
connection = sqlite3.connect(database, check_same_thread=False)
 
# Schema mapping for related words
schema_mapping = {
    "customers": ["clients", "buyers", "patrons"],
    "orders": ["purchases", "transactions", "order records"],
    "products": ["items", "goods", "merchandise", "active products"],
    "categories": ["groups", "types", "product categories"],
    "shippers": ["delivery services", "shipping companies"],
    "orderDate": ["purchase date", "order placed date"],
    "city": ["location", "town", "place"],
    "country": ["nation", "region"],
    "history":["product price history","price history","old price","past history","past record"]
}
 
# Expand schema mapping dynamically using WordNet
def expand_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().lower())
    return synonyms
 
expanded_mapping = {
    key: set(val).union(*(expand_synonyms(v) for v in val))
    for key, val in schema_mapping.items()
}
 
# Preprocess natural language command
def preprocess_command(command):
    words = nltk.word_tokenize(command.lower())
    filtered_words = [word for word in words if word not in stopwords.words('english')]
    return filtered_words
 
# Map related words to schema elements
def map_to_schema(tokens, schema_mapping):
    mapped_elements = {}
    for token in tokens:
        for key, synonyms in schema_mapping.items():
            if token in synonyms or token == key:
                mapped_elements[token] = key
                break
    return mapped_elements
 
def extract_first_sql_query(prompt):
    sql_pattern = re.compile(r"(?s) sql\s*([\s\S]*?\b(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|WITH)\b[\s\S]*?;) |"
        r"\b(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|WITH)\b[\s\S]*?;", re.DOTALL | re.IGNORECASE)
    match = sql_pattern.search(prompt)
    if match:
        return re.sub(r"", "", match.group(0)).strip()
    return "No SQL query found."
 
def generate_query(user_command):
    # Preprocess the command
    tokens = preprocess_command(user_command)
 
    # Map tokens to schema elements
    mapped_elements = map_to_schema(tokens, expanded_mapping)
 
    # Replace natural language terms with SQL-compatible terms
    for natural_term, sql_term in mapped_elements.items():
        user_command = user_command.replace(natural_term, sql_term)
 
    # Set the prompt to guide the model to use the correct tables
    messages = [
        SystemMessage(content=""" 
        Your task is to generate SQL queries for an SQLite database with these tables and columns:
        - **orders**:
          - orderID (INTEGER): Unique identifier for each order
          - customerID (INTEGER): The customer who placed the order
          - orderDate (TEXT): The date when the order was placed
          - requiredDate (TEXT): The date when the customer requested the order to be delivered
          - shippedDate (TEXT): The date when the order was shipped
          - shipperID (INTEGER): The ID of the shipping company used for the order
          - freight (REAL): The shipping cost for the order (USD)
        - **order_details**:
          - orderID (INTEGER): The ID of the order this detail belongs to
          - productID (INTEGER): The ID of the product being ordered
          - unitPrice (REAL): The price per unit of the product at the time the order was placed (USD - discount not included)
          - quantity (INTEGER): The number of units being ordered
          - discount (REAL): The discount percentage applied to the price per unit
        - **customers**:
          - customerID (INTEGER): Unique identifier for each customer
          - companyName (TEXT): The name of the customer's company
          - contactName (TEXT): The name of the primary contact for the customer
          - contactTitle (TEXT): The job title of the primary contact for the customer
          - city (TEXT): The city where the customer is located
          - country (TEXT): The country where the customer is located
        - **products**:
          - productID (INTEGER): Unique identifier for each product
          - productName (TEXT): The name of the product
          - quantityPerUnit (TEXT): The quantity of the product per package
          - unitPrice (REAL): The current price per unit of the product (USD)
          - discontinued (INTEGER): Indicates with a 1 if the product has been discontinued
          - categoryID (INTEGER): The ID of the category the product belongs to
          - start_date (DATE) : The date from which the associated unitPrice becomes valid. It marks the beginning of the price's validity period.
          - end_date (DATE) : The date on which the associated unitPrice stops being valid. If null, the price is considered valid indefinitely.
          - is_active (BOOLEAN) :  A flag indicating whether the associated unitPrice is the currently active price (TRUE for active, FALSE for inactive).
        - **history**:
          - historyID (INTEGER): Unique identifier for each historical price record.
          - productID (INTEGER): Foreign key referencing the productID in the products table, identifying the product associated with the price record.
          - unitPrice (DECIMAL): The price per unit of the product during the specified validity period (in USD).
          - start_date (DATE): The date when the historical price became effective.
          - end_date (DATE, Nullable): The date when the historical price was replaced by a new price. If NULL, the price is valid indefinitely.
        - **categories**:
          - categoryID (INTEGER): Unique identifier for each product category
          - categoryName (TEXT): The name of the category
          - description (TEXT): A description of the category and its products
        - **shippers**:
          - shipperID (INTEGER): Unique identifier for each shipper
          - companyName (TEXT): The name of the company that provides shipping services
        When the user's query refers to specific records, use appropriate filtering conditions like `WHERE`, `AND`, `OR`, or `LIKE`. Ensure that all queries include conditions to fetch only the necessary data. Example: 
          - "Get all orders from customers in London" -> `SELECT * FROM orders WHERE customerID IN (SELECT customerID FROM customers WHERE city = 'London');`
          - "Show products in the 'Beverages' category" -> `SELECT * FROM products WHERE categoryID IN (SELECT categoryID FROM categories WHERE categoryName = 'Beverages');`
        Ensure that generated queries strictly use these column names without creating new ones. Also, maintain the exact case sensitivity of each column and table name.
        """),
        HumanMessage(content=user_command),
    ]
 
    # Invoke the model with the messages
    ai_message = model.invoke(messages)
    print("AI Response:", ai_message.content)
 
    # Extract the SQL query using the helper function
    sql_query = extract_first_sql_query(ai_message.content)
 
    if not sql_query:
        print("Failed to generate SQL query.")
        return
 
    print("Generated SQL Query:", sql_query)
 
    # Execute the SQL query and return results as a DataFrame
    try:
        df = pd.read_sql_query(sql_query, connection)
        print(df.to_string(index=False))  # Display the query result as a table
    except Exception as e:
        print(f"Error executing query: {str(e)}")
 
 
if __name__ == '__main__':
    # Get user input from terminal
    user_command = input("Enter your query: ")
    generate_query(user_command)