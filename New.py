from flask import Flask, request, Response
from joblib import load

#Loading the model
my_lr_model=load("model/my_linear_regression_model.joblib")

from flask import Flask

app = Flask(__name__)

@app.route( "/get_salary_predictions",methods=['POST','GET'])
def test():
    data=request.json
    user_sent_this_data=data.get('mydata')

    user_number=float(user_sent_this_data)
    model_predictions=my_lr_model.predict([[user_number]])
    my_prediction=model_predictions[0]
    return Response(str(my_prediction))

if __name__=='__main__':
    app.run(debug=True)

