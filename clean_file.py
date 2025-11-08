"""
Simple fix for streamlit_app.py encoding issues
"""

def clean_file():
    try:
        # Read the file with UTF-8 encoding
        with open('streamlit_app.py', 'r', encoding='utf-8-sig') as f:
            content = f.read()
        
        # Clean up any problematic BOM characters
        content = content.replace('\ufeff', '')  # Remove BOM
        
        # Save with clean UTF-8 encoding
        with open('streamlit_app.py', 'w', encoding='utf-8', newline='\n') as f:
            f.write(content)
        
        print("✅ File cleaned and saved successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    clean_file()