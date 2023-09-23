from googletrans import Translator

def translate_text(text, source_language, target_language):
    try:
        translator = Translator()
        translation = translator.translate(text, src=source_language, dest=target_language)
        print(translation.text)  # Add this line to check the translated text
        return translation.text
    except Exception as e:
        print(str(e))
        return str(e)


