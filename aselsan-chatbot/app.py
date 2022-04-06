from flask import Flask, render_template, request, jsonify
import re
# import model

app = Flask(__name__, static_url_path="/static")



def show_predictions(text):

  tokenized_sent = tokenizer.encode(text)

  predicted_slots, predicted_intents = joint_model.predict([tokenized_sent])

  intent = le.inverse_transform([np.argmax(predicted_intents)])
  print("="*5, "INTENT", "="*5)
  print(intent)

  slots = np.argmax(predicted_slots, axis=-1)

  slots = [index_to_word[w_idx] for w_idx in slots[0]]

  print("\n")
  print("="*5, "SLOTS","="*5)
  for w,l in zip(tokenizer.tokenize(text),slots[1:-1]):
    print(w,  l)
    
    
  return intent[0], tokenizer.tokenize(text),slots[1:-1]
    
@app.route('/message', methods = ['POST'])
def reply():
    question = request.form["msg"]      
    intent , tokens, slots = show_predictions(question)
    return jsonify( { 'text': [str(intent),' '.join(tokens),' '.join(slots)], 'reload' : False }) 
     

@app.route('/postmethod', methods = ['POST']) 
def post_javascript_data():       
   return ''
    
@app.route("/")
def index():
    return render_template("index.html")

if (__name__ == "__main__"):         
    app.run(host="localhost", port = 5008)

