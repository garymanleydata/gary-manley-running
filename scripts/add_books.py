import requests
import os
from datetime import datetime
import sys

# Setup Paths
vScriptDir = os.path.dirname(os.path.abspath(__file__))
vReviewsDir = os.path.join(vScriptDir, '../reviews')
os.makedirs(vReviewsDir, exist_ok=True)

def search_google_books(query):
    print(f"Searching for '{query}'...")
    url = f"https://www.googleapis.com/books/v1/volumes?q={query}"
    response = requests.get(url)
    if response.status_code != 200:
        print("Error connecting to Google Books.")
        return None
    
    data = response.json()
    if 'items' not in data:
        print("No books found.")
        return None
        
    # Return top result
    return data['items'][0]['volumeInfo']

def create_review_file(book_info):
    title = book_info.get('title', 'Unknown Title')
    authors = ", ".join(book_info.get('authors', ['Unknown Author']))
    
    # Try to get the best image
    image_links = book_info.get('imageLinks', {})
    # Prefer 'thumbnail', fallback to smallThumbnail
    image_url = image_links.get('thumbnail', image_links.get('smallThumbnail', ''))
    
    # Clean up filename
    safe_title = "".join([c if c.isalnum() else "-" for c in title.lower()])
    filename = f"{safe_title}.qmd"
    filepath = os.path.join(vReviewsDir, filename)
    
    # User Inputs
    print(f"\nFOUND: {title} by {authors}")
    rating = input("Star Rating (e.g. ⭐⭐⭐⭐): ")
    category_input = input("Categories (comma separated, e.g. tech, bio): ")
    categories = [c.strip() for c in category_input.split(',')]
    
    # Create the Content
    content = f"""---
title: "{title}"
author: "{authors}"
date: "{datetime.now().strftime('%Y-%m-%d')}"
categories: {categories}
image: "{image_url}"
rating: "{rating}"
description: "Review of {title}"
---

## Summary
*Enter a quick summary here...*

## Key Takeaways
* Point 1
* Point 2

## My Review
Write your full review here...
"""
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\nSUCCESS: Created {filepath}")

if __name__ == "__main__":
    q = input("Enter Book Name or ISBN: ")
    if q:
        book = search_google_books(q)
        if book:
            create_review_file(book)