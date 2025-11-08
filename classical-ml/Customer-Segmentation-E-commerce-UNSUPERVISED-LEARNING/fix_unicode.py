"""
Find and fix Unicode characters in streamlit_app.py
"""
import codecs

def fix_unicode_issues():
    # Read with different encodings to identify the issue
    try:
        with open('streamlit_app.py', 'r', encoding='utf-8') as f:
            content = f.read()
        print("✅ UTF-8 encoding works")
        
        # Find problematic characters
        problematic_chars = []
        for i, char in enumerate(content):
            try:
                char.encode('cp1252')
            except UnicodeEncodeError:
                problematic_chars.append((i, char, ord(char)))
        
        if problematic_chars:
            print(f"Found {len(problematic_chars)} problematic characters:")
            for pos, char, code in problematic_chars[:5]:  # Show first 5
                print(f"Position {pos}: '{char}' (U+{code:04X})")
            
            # Fix by replacing problematic characters
            fixed_content = content
            for pos, char, code in problematic_chars:
                if code == 0x8f:  # Specific problematic character
                    fixed_content = fixed_content.replace(char, '🎯')
                    print(f"Replaced character at position {pos}")
            
            # Write back with UTF-8 BOM to ensure compatibility
            with open('streamlit_app.py', 'w', encoding='utf-8-sig') as f:
                f.write(fixed_content)
            print("✅ File fixed and saved with UTF-8 BOM")
        else:
            print("✅ No problematic characters found")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    fix_unicode_issues()