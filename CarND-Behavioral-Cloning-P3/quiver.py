from quiver_engine import server
from keras.models import load_model
model = load_model("./model.h5")
server.launch(model, input_folder='./imgs')
