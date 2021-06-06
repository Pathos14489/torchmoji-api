from __future__ import print_function, division, unicode_literals
import json
import datetime
from multiprocessing import Value

import numpy as np

from torchmoji.sentence_tokenizer import SentenceTokenizer
from torchmoji.model_def import torchmoji_emojis
from torchmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH

from flask import Flask, request
api = Flask(__name__)

# Emoji map in emoji_overview.png
EMOJIS = ":joy: :unamused: :weary: :sob: :heart_eyes: \
:pensive: :ok_hand: :blush: :heart: :smirk: \
:grin: :notes: :flushed: :100: :sleeping: \
:relieved: :relaxed: :raised_hands: :two_hearts: :expressionless: \
:sweat_smile: :pray: :confused: :kissing_heart: :heartbeat: \
:neutral_face: :information_desk_person: :disappointed: :see_no_evil: :tired_face: \
:v: :sunglasses: :rage: :thumbsup: :cry: \
:sleepy: :yum: :triumph: :hand: :mask: \
:clap: :eyes: :gun: :persevere: :smiling_imp: \
:sweat: :broken_heart: :yellow_heart: :musical_note: :speak_no_evil: \
:wink: :skull: :confounded: :smile: :stuck_out_tongue_winking_eye: \
:angry: :no_good: :muscle: :facepunch: :purple_heart: \
:sparkling_heart: :blue_heart: :grimacing: :sparkles:".split(' ')
counter = Value('i', 0)

def top_elements(array, k):
    ind = np.argpartition(array, -k)[-k:]
    return ind[np.argsort(array[ind])][::-1]

@api.route('/prompt', methods=['POST'])
def main():
    counter.value += 1
    time_start = datetime.datetime.now()
    data = request.get_json(force=True)
    print(data)
    prompt = data['prompt']
    # Running predictions
    tokenized, _, _ = st.tokenize_sentences([prompt])
    # Get sentence probability
    prob = model(tokenized)[0]

    # Top emoji id
    emoji_ids = top_elements(prob, 5)

    # map to emojis
    emojis = map(lambda x: EMOJIS[x], emoji_ids)
    time_end = datetime.datetime.now()
    elapsed_time = time_end - time_start

    output = {
        "requestIndex":counter.value,
        "prompt":prompt,
        "generationTime":elapsed_time.total_seconds() * 1000,
        "emotes":' '.join(emojis),
    }

    print(output)
    return json.dumps(output)

if __name__ == "__main__":
    # Tokenizing using dictionary
    with open(VOCAB_PATH, 'r') as f:
        vocabulary = json.load(f)

    st = SentenceTokenizer(vocabulary, 512) #512 is the max length, not really sure how long it can go, but I don't think I need too long for my purposes. Feel free to tweak

    model = torchmoji_emojis(PRETRAINED_PATH) # Loading model
    api.run(host='0.0.0.0', port=5005)