import os
import re
import inspect
import pytimize

from shutil import rmtree
from collections import deque
from bs4 import BeautifulSoup

def generate():
    style = """
        div {
            width: 80%;
            max-width: 1000px;
            margin: auto;
        }

        pre {
            display: inline-block;
            background-color: #ccc;
            padding: 4px 6px;
        }
    """
    html_mapping = [
        (r"<(.*?)>", r"{\g<1>}"),
        (r"`(.*?)`", r"<pre>\g<1></pre>")
    ]
    folder = os.path.dirname(os.path.realpath(__file__))
    root = os.path.join(folder, "pytimize")

    if os.path.isdir(root):
        rmtree(root)

    os.mkdir(root)

    queue = deque([(pytimize, os.path.join(folder, root))])

    while len(queue) > 0:
        current, path = queue.popleft()

        if inspect.ismodule(current):
            for i in current.__all__:
                next_path = path
                module = getattr(current, i)
                
                if inspect.ismodule(module): 
                    next_path = os.path.join(path, i)

                    os.mkdir(next_path)

                queue.append((module, next_path))
        elif inspect.isclass(current):
            members = filter(lambda x: not x[0].startswith("_") and x[1].__doc__ is not None, inspect.getmembers(current))
            compiled = []
            
            for name, function in members:
                inner = function.__doc__

                for expression, tag in html_mapping:
                    inner = re.sub(expression, tag, inner)

                inner = "<br>".join(map(lambda x: x.strip(), filter(None, inner.split("\n"))))

                rendered = f'''
                    <div>
                        <h2>{name}</h2>
                        <hr>
                        <p>{inner}</p>
                    </div>
                '''

                compiled.append(rendered)

            compiled = "".join(compiled)
            
            if compiled == "":
                continue

            with open(os.path.join(path, f"{current.__name__}.html"), "w") as document:
                html = f'''
                    <html>
                        <head>
                            <title>Pytimize</title>
                            <style>
                                {style}
                            </style>
                        </head>
                        <body>
                            <h1>{current.__name__}</h1>
                            <section>{compiled}</section>
                        </body>
                    </html>
                '''
                soup = BeautifulSoup(html, 'html.parser')

                document.write(soup.prettify())

if __name__ == "__main__":
    generate()
