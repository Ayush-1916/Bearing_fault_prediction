from flask import Flask, request, render_template
import os
from NEW import process_and_plot

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        test_no = request.form.get("Test_no")
        bearing_no = request.form.get("Bearing_no")
        #/Users/ayushkoge/Deployment/index.html
        # Call the function to process data and generate the plot
        try:
            fault_status = process_and_plot(test_no, bearing_no)

            process_and_plot(test_no, bearing_no)
            return render_template("index.html", image="static/scatter_plot.png" , fault_status=fault_status)
        except Exception as e:
            return f"Error: {str(e)}"
    
    return render_template("index.html", image=None , fault_status = None)

if __name__ == "__main__":
    app.run(debug=True)