import sys
import os

print("=" * 70)
print("IMPORT DIAGNOSTIC")
print("=" * 70)

print(f"\n1. CURRENT DIRECTORY: {os.getcwd()}")
print(f"2. PYTHON EXECUTABLE: {sys.executable}")
print(f"3. PYTHON VERSION: {sys.version.split()[0]}")

print("\n4. CHECKING SRC DIRECTORY:")
src_path = os.path.join(os.getcwd(), 'src')
print(f"   Path: {src_path}")
print(f"   Exists: {os.path.exists(src_path)}")
print(f"   Is directory: {os.path.isdir(src_path)}")

print("\n5. CHECKING __init__.py:")
init_path = os.path.join(src_path, '__init__.py')
print(f"   Path: {init_path}")
print(f"   Exists: {os.path.exists(init_path)}")
if os.path.exists(init_path):
    print(f"   Size: {os.path.getsize(init_path)} bytes")
    with open(init_path, 'r') as f:
        content = f.read(100)
        print(f"   First 100 chars: {repr(content)}")

print("\n6. CHECKING LOGGER.PY:")
logger_path = os.path.join(src_path, 'logger.py')
print(f"   Path: {logger_path}")
print(f"   Exists: {os.path.exists(logger_path)}")

print("\n7. PYTHON PATH (first 5 entries):")
for i, path in enumerate(sys.path[:5]):
    print(f"   [{i}] {path}")

print("\n8. TESTING IMPORTS:")

# Test 1: Add current directory to path
print("\n   Test 1: Adding cwd to sys.path")
sys.path.insert(0, os.getcwd())
print(f"   Added: {os.getcwd()}")

# Test 2: Try importing src
print("\n   Test 2: Importing src package")
try:
    import src
    print(f"   ✅ src imported")
    print(f"   src.__file__: {getattr(src, '__file__', 'NO FILE')}")
    print(f"   src.__path__: {getattr(src, '__path__', 'NO PATH')}")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Try importing from src
print("\n   Test 3: Importing from src.logger")
try:
    from src.logger import logging
    print("   ✅ src.logger imported successfully")
    print(f"   logging module: {logging}")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
