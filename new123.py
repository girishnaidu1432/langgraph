import streamlit as st
import requests
from bs4 import BeautifulSoup
import PyPDF2
from io import BytesIO
import easyocr
import openai

# ✅ Azure OpenAI API Configuration
openai.api_key = "14560021aaf84772835d76246b53397a"
openai.api_base = "https://amrxgenai.openai.azure.com/"
openai.api_type = 'azure'
openai.api_version = '2024-02-15-preview'
deployment_name = 'gpt'

# Initialize session state for storing extracted data
if "content" not in st.session_state:
    st.session_state.content = ""

# Function to extract HTML content
def extract_web_content(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        text_content = soup.get_text()
        links = [a['href'] for a in soup.find_all('a', href=True)]
        image_urls = [img['src'] for img in soup.find_all('img')]

        pdf_links = [link for link in links if link.endswith('.pdf')]

        return text_content, links, image_urls, pdf_links
    except Exception as e:
        st.error(f"Error extracting web content: {e}")
        return "", [], [], []

# Function to download and extract PDFs using PyPDF2
def download_and_extract_pdfs(pdf_links):
    pdf_texts = []
    
    for pdf_url in pdf_links:
        try:
            response = requests.get(pdf_url)
            pdf_bytes = BytesIO(response.content)

            reader = PyPDF2.PdfReader(pdf_bytes)
            pdf_text = ""

            for page in reader.pages:
                pdf_text += page.extract_text() or ""

            pdf_texts.append(pdf_text)

        except Exception as e:
            st.error(f"Error extracting PDF content: {e}")

    return pdf_texts

# Function to download and extract text from images using EasyOCR
def download_and_process_images(image_urls):
    reader = easyocr.Reader(['en'])
    image_texts = []

    for img_url in image_urls:
        try:
            response = requests.get(img_url)
            img = BytesIO(response.content)

            # Extract text using EasyOCR
            text = reader.readtext(img, detail=0)
            image_texts.append(" ".join(text))
        except Exception as e:
            st.error(f"Error processing image: {e}")

    return image_texts

# Function to recursively visit and extract content from hyperlinks
def extract_hyperlink_content(links, depth=1):
    hyperlink_texts = []

    if depth == 0:
        return hyperlink_texts

    for link in links:
        if not link.startswith("http"):
            continue  # Skip relative URLs

        try:
            text, new_links, images, pdfs = extract_web_content(link)
            pdf_texts = download_and_extract_pdfs(pdfs)
            image_texts = download_and_process_images(images)

            all_content = f"Text: {text}\nPDFs: {pdf_texts}\nImages: {image_texts}"
            hyperlink_texts.append(all_content)

            # Recursively extract content from new links (depth-limited)
            hyperlink_texts += extract_hyperlink_content(new_links, depth - 1)

        except Exception as e:
            st.error(f"Error extracting content from {link}: {e}")

    return hyperlink_texts

# Chatbot function using Azure OpenAI GPT
def chat_with_llm(query, context):
    try:
        response = openai.ChatCompletion.create(
            engine=deployment_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Context: {context}\n\nQuery: {query}"}
            ]
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Error during LLM response generation: {e}"

# Streamlit UI
st.title("Multi-Modal RAG Chatbot with Azure OpenAI and PyPDF2")

# URL input
url = st.text_input("Enter a URL:")
depth = st.slider("Recursion Depth for Hyperlinks:", 1, 3, 1)

if st.button("Fetch Data"):
    if url:
        with st.spinner("Fetching and extracting content..."):
            text, links, img_urls, pdf_links = extract_web_content(url)

            pdf_texts = download_and_extract_pdfs(pdf_links)
            image_texts = download_and_process_images(img_urls)
            hyperlink_texts = extract_hyperlink_content(links, depth)

            # Combine all content into session state
            all_content = f"Text: {text}\nPDFs: {pdf_texts}\nImages: {image_texts}\nHyperlinks: {hyperlink_texts}"
            st.session_state.content = all_content

            st.success("Data fetched successfully!")
    else:
        st.warning("Please enter a valid URL.")

# Chatbot interface
query = st.text_input("Ask a question about the content:")

if st.button("Get Answer"):
    if st.session_state.content and query:
        with st.spinner("Generating response..."):
            answer = chat_with_llm(query, st.session_state.content)
            st.write("### Response:")
            st.write(answer)
    else:
        st.warning("Please fetch content first and enter a query.")
