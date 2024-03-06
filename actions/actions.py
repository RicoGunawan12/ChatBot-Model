# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

from typing import Any, Text, Dict, List
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import requests

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
#
#
nltk.download('stopwords')
nltk.download('punkt')

data = data = [
    {
      "category_id": 4,
      "name": "Modern Coffee Table",
      "description": "Stylish and functional coffee table with sleek design.",
      "price": 65,
      "stock": 82,
      "media_urls": ["https://i5.walmartimages.com/asr/9997fa8d-5f1d-4972-a018-ccee93c1326f_2.324b83b170a6141ffeda22ac0a8fc35f.jpeg"]
    },
    {
      "category_id": 1,
      "name": "Wireless Bluetooth Earbuds",
      "description": "High-quality wireless earbuds with advanced noise cancellation technology.",
      "price": 90,
      "stock": 45,
      "media_urls": ["https://images-na.ssl-images-amazon.com/images/I/71gtHnQGfQL._AC_SL1500_.jpg"]
    },
    {
      "category_id": 2,
      "name": "Men's Slim Fit Shirt",
      "description": "Classic slim-fit shirt made from premium cotton material.",
      "price": 35,
      "stock": 63,
      "media_urls": ["https://spy.com/wp-content/uploads/2021/06/H2H-Mens-Casual-Slim-Fit-Short-Sleeve-T-Shirts-Cotton-Blended-Soft-Lightweight-Crew-Neck.jpg?resize=714"]
    },
    {
      "category_id": 5,
      "name": "LEGO City Police Station",
      "description": "Build and play with this exciting LEGO City Police Station set.",
      "price": 75,
      "stock": 20,
      "media_urls": ["https://i5.walmartimages.com/asr/c52e424a-02fd-4d43-8e4d-8a7e2faa2016.5b376350242e4eac9a59d801a918e988.jpeg"]
    },
    {
      "category_id": 3,
      "name": "The Great Gatsby by F. Scott Fitzgerald",
      "description": "Classic novel set in the Jazz Age, depicting the American Dream.",
      "price": 15,
      "stock": 92,
      "media_urls": ["https://th.bing.com/th/id/OIP.fFaX7nKq5_5gf2nSI3QEUgHaLK?rs=1&pid=ImgDetMain"]
    },
    {
      "category_id": 1,
      "name": "Smartphone Gimbal Stabilizer",
      "description": "Capture smooth and steady videos with this smartphone gimbal stabilizer.",
      "price": 80,
      "stock": 75,
      "media_urls": ["https://th.bing.com/th/id/OIP.yd6SdLf_YfbmiCIdxYSvUAHaHa?rs=1&pid=ImgDetMain"]
    },
    {
      "category_id": 4,
      "name": "Sectional Sofa with Ottoman",
      "description": "Comfortable sectional sofa with ottoman for modern living rooms.",
      "price": 550,
      "stock": 12,
      "media_urls": ["https://i5.walmartimages.com/asr/b94d5171-326b-4605-bf09-36fe2b36c703_4.3fc3ef68b0d385ec2e86a1bceccbea60.jpeg"]
    },
    {
      "category_id": 5,
      "name": "Remote Control Car",
      "description": "Fast and durable remote control car for outdoor adventures.",
      "price": 40,
      "stock": 60,
      "media_urls": ["https://www.sheknows.com/wp-content/uploads/2019/11/71N8zkI7H4L._AC_SL1500_.jpg"]
    },
    {
      "category_id": 2,
      "name": "Women's Winter Coat",
      "description": "Stay warm and stylish with this cozy winter coat for women.",
      "price": 95,
      "stock": 38,
      "media_urls": ["https://th.bing.com/th/id/OIP.hOhA2RNjSIlLGZ6Jg6ITyQHaOy?rs=1&pid=ImgDetMain"]
    },
    {
      "category_id": 3,
      "name": "Educated by Tara Westover",
      "description": "A remarkable memoir about family, sacrifice, and the power of education.",
      "price": 20,
      "stock": 50,
      "media_urls": ["https://i.pinimg.com/originals/0c/3e/2d/0c3e2d79cdf570d6c8dfe2cfdfb8f993.jpg"]
    },
    {
      "category_id": 3,
      "name": "The Catcher in the Rye by J.D. Salinger",
      "description": "A classic coming-of-age novel that has captured the hearts of readers for generations.",
      "price": 18,
      "stock": 25,
      "media_urls": ["https://i0.wp.com/www.raptisrarebooks.com/images/73470/the-catcher-in-the-rye-jd-salinger-first-edition.jpg?fit=1000%2C800&ssl=1"]
    },
    {
      "category_id": 2,
      "name": "Women's Running Shoes",
      "description": "High-performance running shoes designed for comfort and durability.",
      "price": 80,
      "stock": 30,
      "media_urls": ["https://th.bing.com/th/id/OIP.RAk0wTpA1sO5rqsVXz1MZQAAAA?rs=1&pid=ImgDetMain"]
    },
    {
      "category_id": 5,
      "name": "Board Game: Settlers of Catan",
      "description": "Gather resources, build settlements, and trade with other players in this strategic board game.",
      "price": 45,
      "stock": 20,
      "media_urls": ["https://s.catch.com.au/images/product/0030/30413/5e994d28aff64272002148.jpg"]
    },
    {
      "category_id": 1,
      "name": "Wireless Charging Pad",
      "description": "Charge your devices wirelessly with this sleek and efficient charging pad.",
      "price": 30,
      "stock": 50,
      "media_urls": ["https://www.notebookcheck.net/fileadmin/Notebooks/News/_nc3/20171029_grovemade_wireless_charging_pad_01.jpg"]
    },
    {
      "category_id": 4,
      "name": "Bookshelf",
      "description": "Organize your books and display decorative items with this sturdy bookshelf.",
      "price": 120,
      "stock": 15,
      "media_urls": ["https://th.bing.com/th/id/OIP.tZeZMUp6RkSerzI6uo_8rgHaHa?rs=1&pid=ImgDetMain"]
    },
    {
      "category_id": 3,
      "name": "To Kill a Mockingbird by Harper Lee",
      "description": "A powerful story of racial injustice and moral growth set in the American South.",
      "price": 22,
      "stock": 40,
      "media_urls": ["https://cdn11.bigcommerce.com/s-gibnfyxosi/images/stencil/2560w/products/114990/116752/51IXWZzlgSL__41945.1615559130.jpg?c=1"]
    },
    {
      "category_id": 2,
      "name": "Men's Leather Wallet",
      "description": "Sleek and stylish leather wallet with ample space for cards and cash.",
      "price": 50,
      "stock": 35,
      "media_urls": ["https://th.bing.com/th/id/OIP.uyt01zRqhJiIILULYGDrJwHaGd?rs=1&pid=ImgDetMain"]
    },
    {
      "category_id": 5,
      "name": "Puzzle: World Map",
      "description": "Challenge yourself with this intricately designed world map puzzle.",
      "price": 25,
      "stock": 28,
      "media_urls": ["https://i5.walmartimages.com/asr/1b7901c5-41bb-478f-94c3-616440907cde_1.f16ad5ede9bb07a95afc0769e9fa87c4.jpeg"]
    },
    {
      "category_id": 1,
      "name": "Portable Bluetooth Speaker",
      "description": "Take your music anywhere with this compact and powerful Bluetooth speaker.",
      "price": 60,
      "stock": 45,
      "media_urls": ["https://th.bing.com/th/id/OIP.5dmpGAKZ-stTeXpXPpxU5AHaE6?rs=1&pid=ImgDetMain"]
    },
    {
      "category_id": 4,
      "name": "Dining Table Set",
      "description": "Elegant dining table set with chairs, perfect for family gatherings and dinner parties.",
      "price": 350,
      "stock": 10,
      "media_urls": ["https://i5.walmartimages.com/asr/b063ab0e-7d2b-4954-8403-ef49759e8dc7_3.22e0703f2c58f8b5e5b875c23d9f4cc4.jpeg"]
    },
    {
      "category_id": 1,
      "name": "Smartwatch",
      "description": "Stay connected and track your fitness with this stylish smartwatch.",
      "price": 90,
      "stock": 20,
      "media_urls": ["https://www.amazon.in/images/I/6113mS%2BxhyL._SL1500_.jpg"]
    },
    {
      "category_id": 3,
      "name": "Harry Potter and the Sorcerer's Stone by J.K. Rowling",
      "description": "Embark on a magical journey with the first book in the Harry Potter series.",
      "price": 15,
      "stock": 50,
      "media_urls": ["https://tse2.mm.bing.net/th?id=OIP.0Bt9nt9FxPzAPzfB5elxtQHaLH&pid=Api&P=0&h=180"]
    },
    {
      "category_id": 2,
      "name": "Women's Sneakers",
      "description": "Comfortable and stylish sneakers for everyday wear.",
      "price": 50,
      "stock": 30,
      "media_urls": ["https://tse4.mm.bing.net/th?id=OIP.Rf7mAFoiGn2M5lMSG0gxHwHaHa&pid=Api&P=0&h=180"]
    },
    {
      "category_id": 4,
      "name": "Desk Lamp",
      "description": "Illuminate your workspace with this sleek and adjustable desk lamp.",
      "price": 35,
      "stock": 15,
      "media_urls": ["https://tse3.mm.bing.net/th?id=OIP.I4UMJi5JKzeZxMo3zMi8gAHaHa&pid=Api&P=0&h=180"]
    },
    {
      "category_id": 5,
      "name": "RC Drone",
      "description": "Experience aerial adventures with this high-performance remote control drone.",
      "price": 80,
      "stock": 25,
      "media_urls": ["https://i5.walmartimages.com/asr/35b37bc3-bb01-4cfe-941e-b8bb90bac52a.5b3da9daff76dceb238390267bfa6ef1.jpeg"]
    },
    {
      "category_id": 2,
      "name": "Men's Hoodie",
      "description": "Stay cozy and warm with this comfortable men's hoodie.",
      "price": 40,
      "stock": 40,
      "media_urls": ["https://i2.wp.com/cozexs.com/wp-content/uploads/2020/04/Sweatshirt-Men-2019-NEW-Hoodies-Brand-Male-Long-Sleeve-Solid-Hoodie-men-Black-Red-big-size.jpg"]
    },
    {
      "category_id": 1,
      "name": "Wireless Mouse",
      "description": "Enhance your productivity with this ergonomic wireless mouse.",
      "price": 25,
      "stock": 35,
      "media_urls": ["https://tse3.mm.bing.net/th?id=OIP.d2jD6Bx_9-G5YIfDwHWJVgHaHa&pid=Api&P=0&h=180"]
    },
    {
      "category_id": 4,
      "name": "Accent Chair",
      "description": "Add a touch of style to your living space with this elegant accent chair.",
      "price": 120,
      "stock": 10,
      "media_urls": ["https://tse4.mm.bing.net/th?id=OIP.f6OtnbzL8KK0wOUcfEw0UQHaHa&pid=Api&P=0&h=180"]
    },
    {
      "category_id": 5,
      "name": "Remote Control Boat",
      "description": "Enjoy thrilling adventures on the water with this remote control boat.",
      "price": 60,
      "stock": 20,
      "media_urls": ["https://images-na.ssl-images-amazon.com/images/I/81e8son4XcL._SL1500_.jpg"]
    },
    {
      "category_id": 3,
      "name": "The Hobbit by J.R.R. Tolkien",
      "description": "Join Bilbo Baggins on an epic journey in this timeless fantasy novel.",
      "price": 20,
      "stock": 45,
      "media_urls": ["https://tse1.mm.bing.net/th?id=OIP.G_NxXyG1Byk2hGPmo8xhOwHaLQ&pid=Api&P=0&h=180"]
    },
    {
      "category_id": 2,
      "name": "Men's Watch",
      "description": "Elegant and sophisticated men's watch for everyday wear.",
      "price": 120,
      "stock": 15,
      "media_urls": ["https://tse1.mm.bing.net/th?id=OIP.YMPh0xigPC0BUogmYoUg4gHaHa&pid=Api&P=0&h=180"]
    },
    {
      "category_id": 1,
      "name": "Portable Power Bank",
      "description": "Charge your devices on the go with this powerful portable power bank.",
      "price": 30,
      "stock": 50,
      "media_urls": ["https://tse2.mm.bing.net/th?id=OIP.2ASO2qm_0unfycj3C1Z0uAHaG-&pid=Api&P=0&h=180"]
    },
    {
      "category_id": 5,
      "name": "Remote Control Helicopter",
      "description": "Experience the thrill of flying with this remote control helicopter.",
      "price": 40,
      "stock": 25,
      "media_urls": ["https://www.theontek.com/image/cache/catalog/RC%20HELIP%20P2%20VMAX/Rc%20Remote%20Control%20Helicopters%204%20Channel%202.4%20Ghz%20Battery%20Operated%20Flying%20Model2-2000x2000.jpg"]
    },
    {
      "category_id": 3,
      "name": "1984 by George Orwell",
      "description": "A dystopian classic exploring themes of surveillance and government control.",
      "price": 12,
      "stock": 40,
      "media_urls": ["https://imgv2-2-f.scribdassets.com/img/word_document/338240944/original/82f08c539c/1587828916?v=1"]
    },
    {
      "category_id": 4,
      "name": "Bean Bag Chair",
      "description": "Relax in style with this comfortable and versatile bean bag chair.",
      "price": 50,
      "stock": 20,
      "media_urls": ["https://i5.walmartimages.com/asr/6bfe8928-09c5-44b8-9360-2ab3d07d728a_3.c2ee84e3959d659fb66ba26cf6e318ef.jpeg"]
    },
    {
      "category_id": 1,
      "name": "Wireless Headphones",
      "description": "Immerse yourself in music with these high-quality wireless headphones.",
      "price": 80,
      "stock": 30,
      "media_urls": ["https://tse1.mm.bing.net/th?id=OIP.162XgA3LgtEP8dEewA6ETAHaHa&pid=Api&P=0&h=180"]
    },
    {
      "category_id": 2,
      "name": "Women's Sunglasses",
      "description": "Protect your eyes in style with these fashionable women's sunglasses.",
      "price": 35,
      "stock": 25,
      "media_urls": ["http://ohhmymy.com/wp-content/uploads/2015/10/Ray-Ban-Sunglasses-Specials-Summer-2015-Women.jpg"]
    },
    {
      "category_id": 5,
      "name": "RC Monster Truck",
      "description": "Conquer any terrain with this powerful RC monster truck.",
      "price": 70,
      "stock": 15,
      "media_urls": ["https://i5.walmartimages.com/asr/e681316a-362a-4532-b46c-20b380ffe9f6.c68d07c3855e73ebcd1d7bf935ecd31c.jpeg"]
    },
    {
      "category_id": 3,
      "name": "Pride and Prejudice by Jane Austen",
      "description": "A timeless romance novel exploring themes of love, class, and societal expectations.",
      "price": 15,
      "stock": 35,
      "media_urls": ["https://tse4.mm.bing.net/th?id=OIP.Nu7KKhRxKitVEGyTg_HaBgHaLW&pid=Api&P=0&h=180"]
    },
    {
      "category_id": 4,
      "name": "Standing Desk",
      "description": "Stay productive and improve posture with this adjustable standing desk.",
      "price": 250,
      "stock": 10,
      "media_urls": ["https://i0.wp.com/www.startstanding.org/wp-content/uploads/2019/04/SHW-Electric-Standing-Desk-48-Light-Chery-Best-Standing-Desks.jpg?resize=960%2C925&ssl=1"]
    },
    {
      "category_id": 1,
      "name": "Bluetooth Speaker",
      "description": "Enjoy immersive sound with this portable Bluetooth speaker.",
      "price": 60,
      "stock": 20,
      "media_urls": ["http://cdn.mos.cms.futurecdn.net/cXv8Zr7k9UEmi4BFsMK4im.jpg"]
    },
    {
      "category_id": 3,
      "name": "The Lord of the Rings Trilogy by J.R.R. Tolkien",
      "description": "Embark on an epic adventure through Middle-earth with this classic fantasy trilogy.",
      "price": 30,
      "stock": 50,
      "media_urls": ["https://tse1.mm.bing.net/th?id=OIP.kIfx-8GhAWZxwSpLwjPb_AHaLH&pid=Api&P=0&h=180"]
    },
    {
      "category_id": 2,
      "name": "Men's Dress Shoes",
      "description": "Step out in style with these sophisticated men's dress shoes.",
      "price": 80,
      "stock": 30,
      "media_urls": ["https://tse2.mm.bing.net/th?id=OIP.VSNE-ie3wjcktzpljkmSVQHaF7&pid=Api&P=0&h=180"]
    },
    {
      "category_id": 5,
      "name": "Model Train Set",
      "description": "Build and operate your own miniature railway with this model train set.",
      "price": 100,
      "stock": 15,
      "media_urls": ["https://i5.walmartimages.com/asr/61a8d5c7-a858-4879-b911-d931929abe4e_1.53dce806c456ee177b7bf30952e1b2d3.jpeg"]
    },
    {
      "category_id": 4,
      "name": "Floor Lamp",
      "description": "Illuminate your space with this modern and stylish floor lamp.",
      "price": 70,
      "stock": 25,
      "media_urls": ["https://cdn.webshopapp.com/shops/214805/files/316156679/mid-century-brass-dome-floor-lamp.jpg"]
    },
    {
      "category_id": 1,
      "name": "Wireless Keyboard and Mouse Combo",
      "description": "Work efficiently and clutter-free with this wireless keyboard and mouse combo.",
      "price": 45,
      "stock": 40,
      "media_urls": ["https://rukminim1.flixcart.com/image/1664/1664/keyboard/keyboard-and-mouse-combo/z/h/b/logitech-mk320-original-imadc85hzfyasrqa.jpeg?q=90"]
    },
    {
      "category_id": 2,
      "name": "Women's Backpack",
      "description": "Stay organized and stylish on the go with this chic women's backpack.",
      "price": 55,
      "stock": 35,
      "media_urls": ["https://tse3.mm.bing.net/th?id=OIP.p3TS6L6Z-3N29Icuo35o-gHaIF&pid=Api&P=0&h=180"]
    },
    {
      "category_id": 5,
      "name": "Remote Control Plane",
      "description": "Take to the skies with this high-flying remote control plane.",
      "price": 90,
      "stock": 20,
      "media_urls": ["https://tse4.mm.bing.net/th?id=OIP.RNmSjtcEjiZ27rBlYSf3QQHaHa&pid=Api&P=0&h=180"]
    },
    {
      "category_id": 3,
      "name": "The Catcher in the Rye by J.D. Salinger",
      "description": "A classic coming-of-age novel that has captured the hearts of readers for generations.",
      "price": 15,
      "stock": 30,
      "media_urls": ["https://tse1.mm.bing.net/th?id=OIP.YcZjFghZEvIj3bLLvbLTKAAAAA&pid=Api&P=0&h=180"]
    },
    {
      "category_id": 4,
      "name": "Bar Stool Set",
      "description": "Add seating to your kitchen or bar area with this stylish bar stool set.",
      "price": 120,
      "stock": 10,
      "media_urls": ["https://i5.walmartimages.com/asr/e944f093-71c2-433f-b651-1a14408f37b8.3273955f1d76e1eca8394cf9bf016f53.jpeg"]
    }
]


class ActionTellBestProduct(Action):

    def name(self) -> Text:
        return "action_tell_best_product"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(text="Here is the best product in our store:\nhttps://buybox.com/product=\"Nike Shoe\"")

        return []


class ActionTellProduct(Action):

    def name(self) -> Text:
        return "action_tell_product"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # tracker.latest_message.get('text')
        # print(tracker.latest_message.get('text'))
        last_user_message = tracker.latest_message.get('text')

        tokens = word_tokenize(last_user_message)

        stop_words = set(stopwords.words('english'))
        cleaned_tokens = [word for word in tokens if word.lower() not in stop_words]

        cleaned_message = ' '.join(cleaned_tokens)

        # requests.get('http://bkyz2-fmaaa-aaaaa-qaaaq-cai.localhost:4943/product')

        print(cleaned_message)

        descriptions = [item['description'] for item in data]

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(descriptions)

        cleaned_tfidf = vectorizer.transform([cleaned_message])

        similarities = cosine_similarity(cleaned_tfidf, tfidf_matrix)

        top_indices = np.argsort(similarities[0])[-3:][::-1]

        top_products = [data[i] for i in top_indices]

        # Construct the response message
        response = "Here are the top 3 related products for you:\n"
        for product in top_products:
            product_name = product['name']
            product_description = product['description']
            product_price = product['price']
            response += f"Product: {product_name}\nDescription: {product_description}\nPrice: {product_price} ICP\n\n"

        dispatcher.utter_message(text=response)



        # dispatcher.utter_message(text="Here is what do you need in our store:\nhttps://buybox.com/product=\"Red Shoe\"")

        return []