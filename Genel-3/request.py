import requests

def query_lcm_api(text):
    try:
        # Make sure the API is running at this URL
        url = "http://127.0.0.1:5000/query"
        
        # The payload must match what your API expects - a JSON with a "query" field
        payload = {"query": text}
        
        # Send the POST request
        response = requests.post(url, json=payload)
        
        # Check if request was successful
        response.raise_for_status()
        
        # Return the parsed JSON response
        return response.json()
    
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API. Make sure the server is running.")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        return None
    except requests.exceptions.JSONDecodeError:
        print(f"Error: Could not parse JSON response. Response text: {response.text}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

if __name__ == "__main__":
    # Get user input for the query
    query_text = input("Enter your query text: ")
    
    # Call the API
    result = query_lcm_api(query_text)
    
    # Display the result
    if result:
        print("\nQuery Result:")
        print(f"Query: {result['query']}")
        print("\nMatched Results:")
        for match in result['results']:
            print(f"Text: '{match['text']}'")
            print(f"Similarity: {match['similarity']}")
            print(f"Status: {match['status']}")
            print("-" * 40)
        print(f"Processing time: {result['processing_time_sec']} seconds")