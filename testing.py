import google.generativeai as genai

genai.configure(api_key='AIzaSyDBuhqABFVgEmat3cgsjJ8L6Gt-cFNUkkU')  # Replace with your actual API key
model = genai.GenerativeModel('gemini-1.5-flash')
response = model.generate_content(contents="สวัสดี")

# Extract and print the text directly
#text_to_print = response.candidates[0].content
text_to_print = response.parts[0].text
print(text_to_print) 
