import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Streamlit UI


def load_model():
    # Load the model using joblib
    loaded_model = joblib.load('hack1.joblib')
    return loaded_model

model = load_model()

# Streamlit UI
def main():
    st.title("MentAICare (GAD Detection App)")

    # # Dropdown options - modify these according to your dataset
    # options = [0, 1, 2, 3, 4, 5]

    # # Input fields for features
    # restlessness = st.selectbox('Restlessness', options=options)
    # muscle_tension = st.selectbox('Muscle stiffness', options=options)
    # concentration_difficulties = st.selectbox('Difficulty in concentrating or paying attention', options=options)
    # easy_fatigability = st.selectbox('Easily tired or weak', options=options)
    # anxious_affect = st.selectbox('Anxious Affect in Situations', options=options)
    # anxiety_no_situation = st.selectbox('Anxiety Not Associated with Situation', options=options)
    # exaggerated_startle = st.selectbox('Exaggerated startle Response (reacting to sudden loud sounds, movement, or touch)', options=options)
    # worries_cannot_stop = st.selectbox('Worries that cannot be stopped willingly and occur across more than one activity', options=options)
    # frequency_of_worries = st.selectbox('Always having worries about things that do not matter a lot', options=options)
    # hypochondriasis = st.selectbox('Hypochondriasis (Worrying too much that you are or may become seriously ill)', options=options)

    # # Add more input fields for each feature similarly...
    # ...
    
    # Define the labels for the slider values
    labels = {
        0: "No Effect",
        1: "Very Low",
        2: "Low",
        3: "Moderate",
        4: "High",
        5: "Very High"
    }

    # Function to create a slider with labels
    def labeled_slider(title):
        value = st.slider(title, min_value=0, max_value=5, value=0, format="")
        st.write(f"Selected: {labels[value]}")
        return value

    # Creating sliders for each symptom with labels
    restlessness = labeled_slider('Restlessness')
    muscle_tension = labeled_slider('Muscle stiffness')
    concentration_difficulties = labeled_slider('Difficulty in concentrating or paying attention')
    easy_fatigability = labeled_slider('Easily tired or weak')
    anxious_affect = labeled_slider('Anxious Affect in Situations')
    anxiety_no_situation = labeled_slider('Anxiety Not Associated with Situation')
    exaggerated_startle = labeled_slider('Exaggerated startle Response (reacting to sudden loud sounds, movement, or touch)')
    worries_cannot_stop = labeled_slider('Worries that cannot be stopped willingly')
    frequency_of_worries = labeled_slider('Always having worries about things that do not matter a lot')
    hypochondriasis = labeled_slider('Hypochondriasis (Worrying too much that you are or may become seriously ill)')

# You can use these slider values for further processing or display



    # On submit, make prediction
    if st.button('Predict'):
        new_data = pd.DataFrame({
            'Restlessness': [restlessness],
            'Muscle Tension': [muscle_tension],
            'Concentration difficulties': [concentration_difficulties],
            'Easy fatigability': [easy_fatigability],
            'Anxious affect that occurs in certain situations/environments': [anxious_affect],
            'Anxiety not associated with any particular situation': [anxiety_no_situation],
            'Exaggerated tartle response': [exaggerated_startle],
            'Worries that cannot be stopped voluntarily and occur across more than one acctivity': [worries_cannot_stop],
            'Frequency of worries': [frequency_of_worries],
            'Hypochondriasis': [hypochondriasis]
            # Add other features...
        })

        prediction = model.predict(new_data)
        prediction_prob = model.predict_proba(new_data)

        gad_status = 'GAD Detected' if prediction[0] == 1 else 'No GAD Detected'
        st.write(gad_status)

        # Show feature importance if GAD is detected
        if prediction[0] == 1:
            feature_importances = model.feature_importances_
            features = new_data.columns
            importances_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)

            fig, ax = plt.subplots()
            sns.barplot(x='Importance', y='Feature', data=importances_df, ax=ax)
            st.pyplot(fig)

            # Recommendations
            recommendations = [
                "Avoid staying where can trigger anxiety disorder, lifestyle improvement, physical exercise, and eating healthy foods.",
                "Avoid staying where can trigger anxiety disorder, read recommended books on how to overcome GAD, watch videos from recovered patients, eat healthy, lifestyle adjustments, and exercise regularly.",
                "Opt for counselling by our licensed psychologists by clicking on the required button on our App, reading recommended books on how to overcome GAD, interacting with our WhatsApp Chatbots anxiety disorder management aids, watching videos from recovered patients, and exercising regularly.",
                "Counselling by our licensed psychologists via our App for GAD management, interacting with our WhatsApp chatbot anxiety disorder management aids, exercising regularly, and eating healthy foods."
            ]

            gad_prob = prediction_prob[0][1]

            if gad_prob < 0.5:
                st.write(recommendations[0])
            elif 0.5 <= gad_prob < 0.75:
                st.write(recommendations[1])
            elif 0.75 <= gad_prob < 0.9:
                st.write(recommendations[0])
            else:
                st.write(recommendations[3])

if __name__ == '__main__':
    main()
