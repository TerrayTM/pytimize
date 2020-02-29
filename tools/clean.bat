python -c "for file in __import__('pathlib').Path('..').rglob('*.py[co]'): file.unlink()"
python -c "for folder in __import__('pathlib').Path('..').rglob('__pycache__'): folder.rmdir()"
