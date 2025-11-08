"""
Find problematic lines in streamlit_app.py
"""

def find_problematic_lines():
    try:
        with open('streamlit_app.py', 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        problems = []
        
        for i, line in enumerate(lines, 1):
            # Check if line contains emojis outside of strings
            stripped = line.strip()
            
            # Skip empty lines and comments
            if not stripped or stripped.startswith('#'):
                continue
                
            # Check for lines that start with emojis (likely the problem)
            if any(ord(c) > 127 for c in stripped[:5]) and not stripped.startswith('"') and not stripped.startswith("'"):
                # Check if it's not inside a multiline string
                if not any(quote in line for quote in ['"""', "'''"]):
                    problems.append((i, line.rstrip()))
        
        if problems:
            print(f"Found {len(problems)} problematic lines:")
            for line_num, line_content in problems:
                print(f"Line {line_num}: {repr(line_content)}")
        else:
            print("No obvious problematic lines found")
            
        return problems
        
    except Exception as e:
        print(f"Error: {e}")
        return []

if __name__ == "__main__":
    find_problematic_lines()