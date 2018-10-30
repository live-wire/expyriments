from PIL import Image
from flask import Flask, request
import json
from waitress import serve
import base64
app = Flask(__name__)
from flask import render_template
from object_net import getPrediction
import random
cpu = "--"

# @app.route("/")
# def root():
#     # Loads the index.html in templates/
#     return render_template('index.html', message="Hola PR!")

@app.route("/classify", methods=['GET', 'POST'])
def classify():
    print("Request received!")
    content = request.get_json(silent=True)
    # Got image encoded in base 64, need to convert it to png
    blah = content['base64image']
    blah = blah.replace("data:image/png;base64,","")
    blah = blah.replace(" ","+")
    if content:
        # converting base64 image to png
        with open("imageToSave.png", "wb") as fh:
            fh.write(base64.decodebytes(bytes(blah,'utf-8')))
        im = Image.open("imageToSave.png")
        s = str(random.randint(1,100000))+".png"
        im.save("collection/"+s)
        # now we need a jpg image from the available png
        # converting png to jpg
        rgb_im = im.convert('RGB')
        rgb_im.save('imageToSave.jpg')
        # bg = Image.new("RGB", im.size, (255, 255, 255))
        # bg.paste(im, im)
        # bg.save("imageToSave.jpg")
        # Getting prediction for the jpg
        result = getPrediction(filepath="imageToSave.jpg")
        print("PREDICTION:",result)
        extras = {
            'blah1': 'blah2'
        }
        return json.dumps({'status':'OK','prediction':result, 'extras': extras})
    else:
        return json.dumps({"status":"ERROR"})

@app.route("/job")
def job():
    return json.dumps({'status':'OK'})

if __name__ == "__main__":
    serve(app,host='0.0.0.0', port=5001)