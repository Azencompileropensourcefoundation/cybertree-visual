import tensorflow as tf
import numpy as np
import requests

# 1. Duygusal yanıt sorgulama (emotional-reacto.php)
def get_emotional_response(emotion_data):
    """
    Emotion verisini emotional-reacto.php'ye gönderir ve duygusal yanıt alır.
    """
    url = "https://azencompileropensourcefoundation.com/visual-enginnering/emotional-reacto.php"
    response = requests.post(url, json={"emotion": emotion_data})
    return response.json()  # Duygusal yanıtı al

# 2. Duygusal Modelin Kaydedilmesi
def save_emotion_model(emotion_data):
    """
    Duygusal yanıt verisini kaydedecek model oluşturur ve kaydeder.
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(128, activation='relu', input_dim=1))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))  # Pozitif veya Negatif tepki
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    emotion_array = np.array([0 if emotion_data == 'neutral' else 1])  # 0: neutral, 1: happy/positive
    model.fit(emotion_array.reshape(-1, 1), emotion_array, epochs=1, batch_size=1)
    
    model.save("emotion.h5")  # Modeli kaydet
    print("Emotion model saved as 'emotion.h5'.")

# 3. Ana fonksiyon
def main():
    """
    Duygusal yanıt verisini alır, modelle kaydeder.
    """
    # 1. Duygusal yanıt verisini al (emotional-reacto.php)
    emotion = "happy"  # Örnek olarak "happy" belirliyoruz; bunu dışardan gelen veriye göre belirleyebilirsiniz
    print(f"Getting emotional response for emotion: {emotion}...")
    emotional_response = get_emotional_response(emotion)
    print(f"Emotional Response: {emotional_response}")

    # 2. Sistemin davranışını belirleme
    if emotional_response.get("reaction") == "positive":
        print("The system reacts positively to the object.")
    elif emotional_response.get("reaction") == "negative":
        print("The system reacts negatively to the object.")
    else:
        print("The system reacts neutrally to the object.")
    
    # 3. Duygusal modeli kaydetme
    save_emotion_model(emotion)

if __name__ == "__main__":
    main()
