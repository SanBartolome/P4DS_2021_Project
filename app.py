import os
import glob
from flask import Flask, flash, request, redirect, url_for, render_template
from joblib import load
from pydub import AudioSegment
import scipy.io.wavfile as wav
from python_speech_features import mfcc
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {'mp3', 'wav'}

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

music_model = load('music_model.joblib')
genres = {1: 'blues',
             2: 'hiphop',
             3: 'metal',
             4: 'reggae',
             5: 'classical',
             6: 'country',
             7: 'disco',
             8: 'rock',
             9: 'pop'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def feature_extraction(file):
	print('file to extract the features:')
	print(file)
	features=[]
	(sampleRate,data) = wav.read(file)
	mfcc_feature = mfcc(data,sampleRate,
							winlen=0.020,
							appendEnergy = False)
	meanMatrix = mfcc_feature.mean(0)
	for x in meanMatrix:
		features.append(x)
	return features

@app.route("/", methods=["GET","POST"])
def home():
	if request.method == 'POST':
			if 'file' not in request.files:
				flash('No file part')
				return redirect(request.url)
			file = request.files['file']
			if file.filename == '':
				flash('No selected file')
				return redirect(request.url)
			if file:
				if allowed_file(file.filename):
					filename = secure_filename(file.filename)
					folder = os.path.join(app.instance_path, 'uploads')
					os.makedirs(folder, exist_ok=True)
					files = glob.glob(os.path.join(folder,'*'))
					for f in files:
						os.remove(f)
					file.save(os.path.join(folder, secure_filename(filename)))
					return redirect(url_for('predict', name=filename))
				else:
					flash('Incorrect format')
					return redirect(request.url)
	return render_template('home.html')

@app.route("/predict")
def predict():
	filename = request.args['name']
	extension = filename.rsplit('.', 1)[1].lower()
	filepath = os.path.join(os.path.join(app.instance_path, 'uploads'), filename)
	dst = "test.wav"
	if(extension == 'mp3'):
		sound = AudioSegment.from_mp3(filepath)
		sound.export(dst, format="wav")
		audio_file="test.wav"
		audio_feature=feature_extraction(audio_file)
	elif(extension == 'wav'):
		audio_feature=feature_extraction(filepath)
	pred_audio=music_model.predict([audio_feature])
	result = genres[int(pred_audio)-2]
	return render_template('result.html', value=result)

if __name__ == '__main__':
	app.run(debug=False, use_reloader=True)