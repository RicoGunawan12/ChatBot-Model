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

data = [
    {
        "id": "2fvkz-eorwm-ixqky-2tdl3-od7mr-xznnm-2yctw-gia3f-d6rky-fppij-626",
        "store_id": "4tx5v-6j3z7-pcdxn-drc3c-kglne-vdma5-shwrv-rorei-v3t5u-ao6xh-ghe",
        "category_name": "Clothing",
        "name": "Men's Watch",
        "description": "Elegant and sophisticated men's watch for everyday wear.",
        "price": 120,
        "stock": 15,
        "image_url": [
            "https://tse1.mm.bing.net/th?id=OIP.YMPh0xigPC0BUogmYoUg4gHaHa&pid=Api&P=0&h=180"
        ]
    },
    {
        "id": "2uezq-5fvov-6mgzz-3juhm-3g6mo-rquqm-yozdr-oyyis-ofi4k-jj6ad-cnc",
        "store_id": "dfh7u-2a4dz-nvkkd-rwl76-posyj-vrk7u-ipx5r-45n4v-v5wwj-fgywn-wx6",
        "category_name": "3",
        "name": "Bookshelf",
        "description": "Organize your books and display decorative items with this sturdy bookshelf.",
        "price": 120,
        "stock": 15,
        "image_url": [
            "https://th.bing.com/th/id/OIP.tZeZMUp6RkSerzI6uo_8rgHaHa?rs=1&pid=ImgDetMain"
        ]
    },
    {
        "id": "2xq7d-6gatx-cfocp-7davv-zsb2o-t5wd2-wqqfd-7cnxw-nutd6-ti4eq-yoi",
        "store_id": "v5k25-imqf6-fejxl-erd5r-okvkf-xquae-pocsp-l2o65-fauto-lsnqw-3ds",
        "category_name": "Books",
        "name": "To Kill a Mockingbird by Harper Lee",
        "description": "A powerful story of racial injustice and moral growth set in the American South.",
        "price": 22,
        "stock": 40,
        "image_url": [
            "https://cdn11.bigcommerce.com/s-gibnfyxosi/images/stencil/2560w/products/114990/116752/51IXWZzlgSL__41945.1615559130.jpg?c=1"
        ]
    },
    {
        "id": "3fufk-bzs7w-gxbvv-4cc2n-gsvno-hazts-f4uvt-6nce3-nl6il-pjcj3-vh6",
        "store_id": "wnxuy-3zjq6-xxycz-vltok-3cuqo-r2tlx-367kr-ncvqr-qyeki-wzklw-kno",
        "category_name": "3",
        "name": "Sectional Sofa with Ottoman",
        "description": "Comfortable sectional sofa with ottoman for modern living rooms.",
        "price": 550,
        "stock": 12,
        "image_url": [
            "https://i5.walmartimages.com/asr/b94d5171-326b-4605-bf09-36fe2b36c703_4.3fc3ef68b0d385ec2e86a1bceccbea60.jpeg"
        ]
    },
    {
        "id": "3rhgh-odwl4-wodsn-5e55k-iub47-lonxk-pjsm4-xwrxt-p7nbo-mzo57-fzo",
        "store_id": "kwau5-thhp3-ssdz6-suluj-nimgj-qrqvf-yvw7e-kl6zm-2jggg-ww7vx-agm",
        "category_name": "0",
        "name": "RC Drone",
        "description": "Experience aerial adventures with this high-performance remote control drone.",
        "price": 80,
        "stock": 25,
        "image_url": [
            "https://i5.walmartimages.com/asr/35b37bc3-bb01-4cfe-941e-b8bb90bac52a.5b3da9daff76dceb238390267bfa6ef1.jpeg"
        ]
    },
    {
        "id": "563i7-pc6nn-6sn7o-h3upg-n2qdl-tojoe-64ffm-2wcnz-qsy7g-pocdt-rhi",
        "store_id": "v5k25-imqf6-fejxl-erd5r-okvkf-xquae-pocsp-l2o65-fauto-lsnqw-3ds",
        "category_name": "Books",
        "name": "Educated by Tara Westover",
        "description": "A remarkable memoir about family, sacrifice, and the power of education.",
        "price": 20,
        "stock": 50,
        "image_url": [
            "https://i.pinimg.com/originals/0c/3e/2d/0c3e2d79cdf570d6c8dfe2cfdfb8f993.jpg"
        ]
    },
    {
        "id": "5pa4w-bsbm6-34url-gbjl3-gs3sy-ecplt-2tbaz-dnghe-fhqpl-hdqqy-iac",
        "store_id": "kwau5-thhp3-ssdz6-suluj-nimgj-qrqvf-yvw7e-kl6zm-2jggg-ww7vx-agm",
        "category_name": "0",
        "name": "Wireless Headphones",
        "description": "Immerse yourself in music with these high-quality wireless headphones.",
        "price": 80,
        "stock": 30,
        "image_url": [
            "https://tse1.mm.bing.net/th?id=OIP.162XgA3LgtEP8dEewA6ETAHaHa&pid=Api&P=0&h=180"
        ]
    },
    {
        "id": "5skh4-izlfo-vd2dz-k2dg3-vv543-jh2sr-nbef7-bwt32-c3rwr-2ppw2-wfw",
        "store_id": "v5k25-imqf6-fejxl-erd5r-okvkf-xquae-pocsp-l2o65-fauto-lsnqw-3ds",
        "category_name": "Books",
        "name": "The Catcher in the Rye by J.D. Salinger",
        "description": "A classic coming-of-age novel that has captured the hearts of readers for generations.",
        "price": 15,
        "stock": 30,
        "image_url": [
            "https://tse1.mm.bing.net/th?id=OIP.YcZjFghZEvIj3bLLvbLTKAAAAA&pid=Api&P=0&h=180"
        ]
    },
    {
        "id": "6j3qt-yudax-7hv7h-nj3om-2ztis-fhh6u-cx65e-zjzfm-eodml-bekg7-hxu",
        "store_id": "wnxuy-3zjq6-xxycz-vltok-3cuqo-r2tlx-367kr-ncvqr-qyeki-wzklw-kno",
        "category_name": "0",
        "name": "Wireless Keyboard and Mouse Combo",
        "description": "Work efficiently and clutter-free with this wireless keyboard and mouse combo.",
        "price": 45,
        "stock": 40,
        "image_url": [
            "https://rukminim1.flixcart.com/image/1664/1664/keyboard/keyboard-and-mouse-combo/z/h/b/logitech-mk320-original-imadc85hzfyasrqa.jpeg?q=90"
        ]
    },
    {
        "id": "6uqiz-2ckiy-qq74m-24rjr-pe7xh-carlp-gxtqb-iy2yj-ks2zn-jwq6a-y2a",
        "store_id": "v5k25-imqf6-fejxl-erd5r-okvkf-xquae-pocsp-l2o65-fauto-lsnqw-3ds",
        "category_name": "Books",
        "name": "The Hobbit by J.R.R. Tolkien",
        "description": "Join Bilbo Baggins on an epic journey in this timeless fantasy novel.",
        "price": 20,
        "stock": 45,
        "image_url": [
            "https://tse1.mm.bing.net/th?id=OIP.G_NxXyG1Byk2hGPmo8xhOwHaLQ&pid=Api&P=0&h=180"
        ]
    },
    {
        "id": "6wup3-d7kpp-gs2av-dtqrj-7d5jy-tg523-iqtox-aqrcp-tlqml-ix7un-nh4",
        "store_id": "4tx5v-6j3z7-pcdxn-drc3c-kglne-vdma5-shwrv-rorei-v3t5u-ao6xh-ghe",
        "category_name": "Clothing",
        "name": "Women's Sneakers",
        "description": "Comfortable and stylish sneakers for everyday wear.",
        "price": 50,
        "stock": 30,
        "image_url": [
            "https://tse4.mm.bing.net/th?id=OIP.Rf7mAFoiGn2M5lMSG0gxHwHaHa&pid=Api&P=0&h=180"
        ]
    },
    {
        "id": "7qrbl-pqmei-44uya-jljbs-vmndh-x4nvv-c4s3n-uigbm-ulhwd-xl3nc-xdu",
        "store_id": "dfh7u-2a4dz-nvkkd-rwl76-posyj-vrk7u-ipx5r-45n4v-v5wwj-fgywn-wx6",
        "category_name": "3",
        "name": "Modern Coffee Table",
        "description": "Stylish and functional coffee table with sleek design.",
        "price": 65,
        "stock": 82,
        "image_url": [
            "https://i5.walmartimages.com/asr/9997fa8d-5f1d-4972-a018-ccee93c1326f_2.324b83b170a6141ffeda22ac0a8fc35f.jpeg"
        ]
    },
    {
        "id": "7rx4x-nvv2q-cvg2v-vpug4-ffz3g-w722a-qw4kp-p6v6x-ajpvc-6itgb-rsi",
        "store_id": "dfh7u-2a4dz-nvkkd-rwl76-posyj-vrk7u-ipx5r-45n4v-v5wwj-fgywn-wx6",
        "category_name": "3",
        "name": "Floor Lamp",
        "description": "Illuminate your space with this modern and stylish floor lamp.",
        "price": 70,
        "stock": 25,
        "image_url": [
            "https://cdn.webshopapp.com/shops/214805/files/316156679/mid-century-brass-dome-floor-lamp.jpg"
        ]
    },
    {
        "id": "a6bch-bbjin-jzmrf-54ect-mttbr-7rona-bpa6g-wqn2i-szzkc-k26js-f44",
        "store_id": "kwau5-thhp3-ssdz6-suluj-nimgj-qrqvf-yvw7e-kl6zm-2jggg-ww7vx-agm",
        "category_name": "0",
        "name": "Portable Power Bank",
        "description": "Charge your devices on the go with this powerful portable power bank.",
        "price": 30,
        "stock": 50,
        "image_url": [
            "https://tse2.mm.bing.net/th?id=OIP.2ASO2qm_0unfycj3C1Z0uAHaG-&pid=Api&P=0&h=180"
        ]
    },
    {
        "id": "c6b2l-5e7kg-l2eqv-v7saf-dkce3-gtqer-nggng-xyq2x-vpghw-6yrcv-exq",
        "store_id": "kwau5-thhp3-ssdz6-suluj-nimgj-qrqvf-yvw7e-kl6zm-2jggg-ww7vx-agm",
        "category_name": "0",
        "name": "Bluetooth Speaker",
        "description": "Enjoy immersive sound with this portable Bluetooth speaker.",
        "price": 60,
        "stock": 20,
        "image_url": [
            "http://cdn.mos.cms.futurecdn.net/cXv8Zr7k9UEmi4BFsMK4im.jpg"
        ]
    },
    {
        "id": "ddvnm-2rdbc-qd2d6-deu7a-bvmg4-ugudo-fldma-yn7qy-nfk35-wgbl7-66k",
        "store_id": "v5k25-imqf6-fejxl-erd5r-okvkf-xquae-pocsp-l2o65-fauto-lsnqw-3ds",
        "category_name": "Books",
        "name": "The Lord of the Rings Trilogy by J.R.R. Tolkien",
        "description": "Embark on an epic adventure through Middle-earth with this classic fantasy trilogy.",
        "price": 30,
        "stock": 50,
        "image_url": [
            "https://tse1.mm.bing.net/th?id=OIP.kIfx-8GhAWZxwSpLwjPb_AHaLH&pid=Api&P=0&h=180"
        ]
    },
    {
        "id": "dee7t-ccdnw-5o4ld-iepaz-tpqag-gjkys-ftm67-q4h5j-djhta-svsjj-h3o",
        "store_id": "dfh7u-2a4dz-nvkkd-rwl76-posyj-vrk7u-ipx5r-45n4v-v5wwj-fgywn-wx6",
        "category_name": "3",
        "name": "Dining Table Set",
        "description": "Elegant dining table set with chairs, perfect for family gatherings and dinner parties.",
        "price": 350,
        "stock": 10,
        "image_url": [
            "https://i5.walmartimages.com/asr/b063ab0e-7d2b-4954-8403-ef49759e8dc7_3.22e0703f2c58f8b5e5b875c23d9f4cc4.jpeg"
        ]
    },
    {
        "id": "dpeix-566hj-hwbgw-vihwl-6jspb-i7wlr-tdzbr-rt46u-zzcu6-tnsen-jlc",
        "store_id": "dfh7u-2a4dz-nvkkd-rwl76-posyj-vrk7u-ipx5r-45n4v-v5wwj-fgywn-wx6",
        "category_name": "3",
        "name": "Bean Bag Chair",
        "description": "Relax in style with this comfortable and versatile bean bag chair.",
        "price": 50,
        "stock": 20,
        "image_url": [
            "https://i5.walmartimages.com/asr/6bfe8928-09c5-44b8-9360-2ab3d07d728a_3.c2ee84e3959d659fb66ba26cf6e318ef.jpeg"
        ]
    },
    {
        "id": "eawme-ox5bg-vaqw4-ok2eb-6osib-duh5z-riwra-opqqv-trdp4-2nw6x-xt2",
        "store_id": "kwau5-thhp3-ssdz6-suluj-nimgj-qrqvf-yvw7e-kl6zm-2jggg-ww7vx-agm",
        "category_name": "4",
        "name": "Remote Control Helicopter",
        "description": "Experience the thrill of flying with this remote control helicopter.",
        "price": 40,
        "stock": 25,
        "image_url": [
            "https://www.theontek.com/image/cache/catalog/RC%20HELIP%20P2%20VMAX/Rc%20Remote%20Control%20Helicopters%204%20Channel%202.4%20Ghz%20Battery%20Operated%20Flying%20Model2-2000x2000.jpg"
        ]
    },
    {
        "id": "ebo37-tv5xi-5cjae-hindn-54hby-ggp7t-p6wb7-m2izm-fhutq-lhqfb-jmw",
        "store_id": "wnxuy-3zjq6-xxycz-vltok-3cuqo-r2tlx-367kr-ncvqr-qyeki-wzklw-kno",
        "category_name": "0",
        "name": "Wireless Bluetooth Earbuds",
        "description": "High-quality wireless earbuds with advanced noise cancellation technology.",
        "price": 90,
        "stock": 45,
        "image_url": [
            "https://images-na.ssl-images-amazon.com/images/I/71gtHnQGfQL._AC_SL1500_.jpg"
        ]
    },
    {
        "id": "egedi-z4gdg-c7liq-p5xc4-i7jlg-lhvlx-aolbp-cg6gu-mt7av-yjjqe-wnk",
        "store_id": "4tx5v-6j3z7-pcdxn-drc3c-kglne-vdma5-shwrv-rorei-v3t5u-ao6xh-ghe",
        "category_name": "Clothing",
        "name": "Men's Hoodie",
        "description": "Stay cozy and warm with this comfortable men's hoodie.",
        "price": 40,
        "stock": 40,
        "image_url": [
            "https://i2.wp.com/cozexs.com/wp-content/uploads/2020/04/Sweatshirt-Men-2019-NEW-Hoodies-Brand-Male-Long-Sleeve-Solid-Hoodie-men-Black-Red-big-size.jpg"
        ]
    },
    {
        "id": "fsocw-yjo5f-7k6vs-jqhvq-3mfbd-bblh5-4a3c3-4nvmf-vhrlq-fzsk5-wlu",
        "store_id": "4tx5v-6j3z7-pcdxn-drc3c-kglne-vdma5-shwrv-rorei-v3t5u-ao6xh-ghe",
        "category_name": "Clothing",
        "name": "Women's Winter Coat",
        "description": "Stay warm and stylish with this cozy winter coat for women.",
        "price": 95,
        "stock": 38,
        "image_url": [
            "https://th.bing.com/th/id/OIP.hOhA2RNjSIlLGZ6Jg6ITyQHaOy?rs=1&pid=ImgDetMain"
        ]
    },
    {
        "id": "h2hiq-lvvyf-76zwu-k6w6t-5yg5t-hspsh-5wz3u-iybxd-nbay6-arotq-lqs",
        "store_id": "wnxuy-3zjq6-xxycz-vltok-3cuqo-r2tlx-367kr-ncvqr-qyeki-wzklw-kno",
        "category_name": "4",
        "name": "LEGO City Police Station",
        "description": "Build and play with this exciting LEGO City Police Station set.",
        "price": 75,
        "stock": 20,
        "image_url": [
            "https://i5.walmartimages.com/asr/c52e424a-02fd-4d43-8e4d-8a7e2faa2016.5b376350242e4eac9a59d801a918e988.jpeg"
        ]
    },
    {
        "id": "h4mql-cyxqe-fgght-qfltp-moabs-dyupe-xvnep-zk6fl-f44bz-e25xb-5oq",
        "store_id": "wnxuy-3zjq6-xxycz-vltok-3cuqo-r2tlx-367kr-ncvqr-qyeki-wzklw-kno",
        "category_name": "4",
        "name": "Remote Control Car",
        "description": "Fast and durable remote control car for outdoor adventures.",
        "price": 40,
        "stock": 60,
        "image_url": [
            "https://www.sheknows.com/wp-content/uploads/2019/11/71N8zkI7H4L._AC_SL1500_.jpg"
        ]
    },
    {
        "id": "h663c-qnof5-g7gue-6uufn-lzla7-xdpuk-uldlz-yb2pw-sbmey-6lamc-av2",
        "store_id": "4tx5v-6j3z7-pcdxn-drc3c-kglne-vdma5-shwrv-rorei-v3t5u-ao6xh-ghe",
        "category_name": "Clothing",
        "name": "Women's Running Shoes",
        "description": "High-performance running shoes designed for comfort and durability.",
        "price": 80,
        "stock": 30,
        "image_url": [
            "https://th.bing.com/th/id/OIP.RAk0wTpA1sO5rqsVXz1MZQAAAA?rs=1&pid=ImgDetMain"
        ]
    },
    {
        "id": "hfu73-xcqpu-a7aji-k7ecd-2otfn-beiqc-3jyld-5ox3s-inqnp-hlkpm-yhq",
        "store_id": "4tx5v-6j3z7-pcdxn-drc3c-kglne-vdma5-shwrv-rorei-v3t5u-ao6xh-ghe",
        "category_name": "Clothing",
        "name": "Men's Dress Shoes",
        "description": "Step out in style with these sophisticated men's dress shoes.",
        "price": 80,
        "stock": 30,
        "image_url": [
            "https://tse2.mm.bing.net/th?id=OIP.VSNE-ie3wjcktzpljkmSVQHaF7&pid=Api&P=0&h=180"
        ]
    },
    {
        "id": "hr224-uq6du-z7krt-qdsa3-hdrcb-7g4x3-6fkys-rdnou-fhq7z-jrwp4-6hq",
        "store_id": "dfh7u-2a4dz-nvkkd-rwl76-posyj-vrk7u-ipx5r-45n4v-v5wwj-fgywn-wx6",
        "category_name": "3",
        "name": "Accent Chair",
        "description": "Add a touch of style to your living space with this elegant accent chair.",
        "price": 120,
        "stock": 10,
        "image_url": [
            "https://tse4.mm.bing.net/th?id=OIP.f6OtnbzL8KK0wOUcfEw0UQHaHa&pid=Api&P=0&h=180"
        ]
    },
    {
        "id": "i76wh-wjblj-7wyo2-ojlt3-b6hjl-ofrxs-pv4tu-mfkzj-j65eh-lldj4-4vy",
        "store_id": "kwau5-thhp3-ssdz6-suluj-nimgj-qrqvf-yvw7e-kl6zm-2jggg-ww7vx-agm",
        "category_name": "4",
        "name": "Remote Control Boat",
        "description": "Enjoy thrilling adventures on the water with this remote control boat.",
        "price": 60,
        "stock": 20,
        "image_url": [
            "https://images-na.ssl-images-amazon.com/images/I/81e8son4XcL._SL1500_.jpg"
        ]
    },
    {
        "id": "jlsdu-vx6dq-2c7uj-klhly-66e3g-thaqa-rwt62-tv6s6-tfqmd-uyubp-rry",
        "store_id": "dfh7u-2a4dz-nvkkd-rwl76-posyj-vrk7u-ipx5r-45n4v-v5wwj-fgywn-wx6",
        "category_name": "4",
        "name": "Puzzle: World Map",
        "description": "Challenge yourself with this intricately designed world map puzzle.",
        "price": 25,
        "stock": 28,
        "image_url": [
            "https://i5.walmartimages.com/asr/1b7901c5-41bb-478f-94c3-616440907cde_1.f16ad5ede9bb07a95afc0769e9fa87c4.jpeg"
        ]
    },
    {
        "id": "l55x7-t4dlj-7yg26-lhzfx-rj7g6-rln7b-3kzjj-gw3si-6qln2-rg4co-waa",
        "store_id": "kwau5-thhp3-ssdz6-suluj-nimgj-qrqvf-yvw7e-kl6zm-2jggg-ww7vx-agm",
        "category_name": "4",
        "name": "RC Monster Truck",
        "description": "Conquer any terrain with this powerful RC monster truck.",
        "price": 70,
        "stock": 15,
        "image_url": [
            "https://i5.walmartimages.com/asr/e681316a-362a-4532-b46c-20b380ffe9f6.c68d07c3855e73ebcd1d7bf935ecd31c.jpeg"
        ]
    },
    {
        "id": "lhlb6-pqczl-qmgkp-q6azo-gbb6t-cj7hk-4ymls-sa5gf-zoidr-you64-rjo",
        "store_id": "wnxuy-3zjq6-xxycz-vltok-3cuqo-r2tlx-367kr-ncvqr-qyeki-wzklw-kno",
        "category_name": "3",
        "name": "Bar Stool Set",
        "description": "Add seating to your kitchen or bar area with this stylish bar stool set.",
        "price": 120,
        "stock": 10,
        "image_url": [
            "https://i5.walmartimages.com/asr/e944f093-71c2-433f-b651-1a14408f37b8.3273955f1d76e1eca8394cf9bf016f53.jpeg"
        ]
    },
    {
        "id": "mctyb-vb2np-jryqa-zk34u-2abo2-vuhj4-pcxny-mywqx-6h4vu-o5do2-w3y",
        "store_id": "wnxuy-3zjq6-xxycz-vltok-3cuqo-r2tlx-367kr-ncvqr-qyeki-wzklw-kno",
        "category_name": "4",
        "name": "Remote Control Plane",
        "description": "Take to the skies with this high-flying remote control plane.",
        "price": 90,
        "stock": 20,
        "image_url": [
            "https://tse4.mm.bing.net/th?id=OIP.RNmSjtcEjiZ27rBlYSf3QQHaHa&pid=Api&P=0&h=180"
        ]
    },
    {
        "id": "mknsv-iprvt-uiakd-fibkh-gpirb-vbuny-sffph-7g6mv-7agxq-zpadh-dce",
        "store_id": "4tx5v-6j3z7-pcdxn-drc3c-kglne-vdma5-shwrv-rorei-v3t5u-ao6xh-ghe",
        "category_name": "Clothing",
        "name": "Women's Backpack",
        "description": "Stay organized and stylish on the go with this chic women's backpack.",
        "price": 55,
        "stock": 35,
        "image_url": [
            "https://tse3.mm.bing.net/th?id=OIP.p3TS6L6Z-3N29Icuo35o-gHaIF&pid=Api&P=0&h=180"
        ]
    },
    {
        "id": "nqsfh-mkbbi-6jcp4-ss3px-bsxnb-zlalq-ahxjg-4qaxr-js6p6-5bflg-akg",
        "store_id": "kwau5-thhp3-ssdz6-suluj-nimgj-qrqvf-yvw7e-kl6zm-2jggg-ww7vx-agm",
        "category_name": "0",
        "name": "Smartwatch",
        "description": "Stay connected and track your fitness with this stylish smartwatch.",
        "price": 90,
        "stock": 20,
        "image_url": [
            "https://www.amazon.in/images/I/6113mS%2BxhyL._SL1500_.jpg"
        ]
    },
    {
        "id": "nvauh-ow22j-wdd4u-pvaod-pjwgf-d2lc2-chz5h-vga5h-2pwmn-qqx3e-m2m",
        "store_id": "v5k25-imqf6-fejxl-erd5r-okvkf-xquae-pocsp-l2o65-fauto-lsnqw-3ds",
        "category_name": "Books",
        "name": "The Catcher in the Rye by J.D. Salinger",
        "description": "A classic coming-of-age novel that has captured the hearts of readers for generations.",
        "price": 18,
        "stock": 25,
        "image_url": [
            "https://i0.wp.com/www.raptisrarebooks.com/images/73470/the-catcher-in-the-rye-jd-salinger-first-edition.jpg?fit=1000%2C800&ssl=1"
        ]
    },
    {
        "id": "o22zi-ql2i4-wg235-thlwd-cirpv-vjzjt-klpk7-nsb55-drbn4-rxsdc-hwu",
        "store_id": "dfh7u-2a4dz-nvkkd-rwl76-posyj-vrk7u-ipx5r-45n4v-v5wwj-fgywn-wx6",
        "category_name": "4",
        "name": "Model Train Set",
        "description": "Build and operate your own miniature railway with this model train set.",
        "price": 100,
        "stock": 15,
        "image_url": [
            "https://i5.walmartimages.com/asr/61a8d5c7-a858-4879-b911-d931929abe4e_1.53dce806c456ee177b7bf30952e1b2d3.jpeg"
        ]
    },
    {
        "id": "pmksc-higwq-4hx7h-qget2-yoadq-dx4nt-kfgh3-dqe6x-kpdpl-lgtp6-4x2",
        "store_id": "wnxuy-3zjq6-xxycz-vltok-3cuqo-r2tlx-367kr-ncvqr-qyeki-wzklw-kno",
        "category_name": "0",
        "name": "Wireless Charging Pad",
        "description": "Charge your devices wirelessly with this sleek and efficient charging pad.",
        "price": 30,
        "stock": 50,
        "image_url": [
            "https://www.notebookcheck.net/fileadmin/Notebooks/News/_nc3/20171029_grovemade_wireless_charging_pad_01.jpg"
        ]
    },
    {
        "id": "r4ivd-vqvo5-ni6eb-fa6ok-b7ack-xbzi7-uxepz-u5zii-u374a-jhgxx-zvk",
        "store_id": "wnxuy-3zjq6-xxycz-vltok-3cuqo-r2tlx-367kr-ncvqr-qyeki-wzklw-kno",
        "category_name": "0",
        "name": "Smartphone Gimbal Stabilizer",
        "description": "Capture smooth and steady videos with this smartphone gimbal stabilizer.",
        "price": 80,
        "stock": 75,
        "image_url": [
            "https://th.bing.com/th/id/OIP.yd6SdLf_YfbmiCIdxYSvUAHaHa?rs=1&pid=ImgDetMain"
        ]
    },
    {
        "id": "rr4ec-pqpcq-oud5b-2irqa-xbgzm-g6fzi-i6kxq-5d4tk-7fuss-keiqs-vym",
        "store_id": "v5k25-imqf6-fejxl-erd5r-okvkf-xquae-pocsp-l2o65-fauto-lsnqw-3ds",
        "category_name": "Books",
        "name": "Harry Potter and the Sorcerer's Stone by J.K. Rowling",
        "description": "Embark on a magical journey with the first book in the Harry Potter series.",
        "price": 15,
        "stock": 50,
        "image_url": [
            "https://tse2.mm.bing.net/th?id=OIP.0Bt9nt9FxPzAPzfB5elxtQHaLH&pid=Api&P=0&h=180"
        ]
    },
    {
        "id": "s3uqz-vsdwc-35iom-rvsp2-xshuc-ysysm-awb5j-bctfb-gwdic-5l5tl-jiu",
        "store_id": "dfh7u-2a4dz-nvkkd-rwl76-posyj-vrk7u-ipx5r-45n4v-v5wwj-fgywn-wx6",
        "category_name": "0",
        "name": "Portable Bluetooth Speaker",
        "description": "Take your music anywhere with this compact and powerful Bluetooth speaker.",
        "price": 60,
        "stock": 45,
        "image_url": [
            "https://th.bing.com/th/id/OIP.5dmpGAKZ-stTeXpXPpxU5AHaE6?rs=1&pid=ImgDetMain"
        ]
    },
    {
        "id": "tsq5p-mysrj-flez2-lzmqf-qgzt3-bcck7-mlbzl-rucf3-63glb-fk2bo-big",
        "store_id": "dfh7u-2a4dz-nvkkd-rwl76-posyj-vrk7u-ipx5r-45n4v-v5wwj-fgywn-wx6",
        "category_name": "3",
        "name": "Standing Desk",
        "description": "Stay productive and improve posture with this adjustable standing desk.",
        "price": 250,
        "stock": 10,
        "image_url": [
            "https://i0.wp.com/www.startstanding.org/wp-content/uploads/2019/04/SHW-Electric-Standing-Desk-48-Light-Chery-Best-Standing-Desks.jpg?resize=960%2C925&ssl=1"
        ]
    },
    {
        "id": "u3bdw-wnvhx-zhmkk-qvicr-pgkt5-5fudp-gintw-6nmqx-dk6wk-774jj-pq6",
        "store_id": "v5k25-imqf6-fejxl-erd5r-okvkf-xquae-pocsp-l2o65-fauto-lsnqw-3ds",
        "category_name": "Books",
        "name": "1984 by George Orwell",
        "description": "A dystopian classic exploring themes of surveillance and government control.",
        "price": 12,
        "stock": 40,
        "image_url": [
            "https://imgv2-2-f.scribdassets.com/img/word_document/338240944/original/82f08c539c/1587828916?v=1"
        ]
    },
    {
        "id": "ue4ex-kvwsv-ztitn-fk66l-yoqdq-6eyuo-d53cw-hgku7-2sdbc-qagfp-65y",
        "store_id": "kwau5-thhp3-ssdz6-suluj-nimgj-qrqvf-yvw7e-kl6zm-2jggg-ww7vx-agm",
        "category_name": "3",
        "name": "Desk Lamp",
        "description": "Illuminate your workspace with this sleek and adjustable desk lamp.",
        "price": 35,
        "stock": 15,
        "image_url": [
            "https://tse3.mm.bing.net/th?id=OIP.I4UMJi5JKzeZxMo3zMi8gAHaHa&pid=Api&P=0&h=180"
        ]
    },
    {
        "id": "v5z6s-wipqw-2wp7s-s6q6f-4sk7q-sx6sw-quks2-lmx3h-q3hxi-xszvr-7ts",
        "store_id": "4tx5v-6j3z7-pcdxn-drc3c-kglne-vdma5-shwrv-rorei-v3t5u-ao6xh-ghe",
        "category_name": "Clothing",
        "name": "Women's Sunglasses",
        "description": "Protect your eyes in style with these fashionable women's sunglasses.",
        "price": 35,
        "stock": 25,
        "image_url": [
            "http://ohhmymy.com/wp-content/uploads/2015/10/Ray-Ban-Sunglasses-Specials-Summer-2015-Women.jpg"
        ]
    },
    {
        "id": "vhe3j-kzgyq-mo25r-xkype-elnmr-4ks6o-73ie7-6ur7n-o6x3g-d3j46-fre",
        "store_id": "v5k25-imqf6-fejxl-erd5r-okvkf-xquae-pocsp-l2o65-fauto-lsnqw-3ds",
        "category_name": "Books",
        "name": "The Great Gatsby by F. Scott Fitzgerald",
        "description": "Classic novel set in the Jazz Age, depicting the American Dream.",
        "price": 15,
        "stock": 92,
        "image_url": [
            "https://th.bing.com/th/id/OIP.fFaX7nKq5_5gf2nSI3QEUgHaLK?rs=1&pid=ImgDetMain"
        ]
    },
    {
        "id": "vvvxw-ytknc-5y45i-v5soe-a5ga4-ryq4c-6gdba-j5rre-idik6-nx76v-uwk",
        "store_id": "kwau5-thhp3-ssdz6-suluj-nimgj-qrqvf-yvw7e-kl6zm-2jggg-ww7vx-agm",
        "category_name": "0",
        "name": "Wireless Mouse",
        "description": "Enhance your productivity with this ergonomic wireless mouse.",
        "price": 25,
        "stock": 35,
        "image_url": [
            "https://tse3.mm.bing.net/th?id=OIP.d2jD6Bx_9-G5YIfDwHWJVgHaHa&pid=Api&P=0&h=180"
        ]
    },
    {
        "id": "yeqvz-g5m4p-5hygq-2tm3j-azeai-dm5tk-zzhcf-zvbpg-ucat2-izatb-sxy",
        "store_id": "v5k25-imqf6-fejxl-erd5r-okvkf-xquae-pocsp-l2o65-fauto-lsnqw-3ds",
        "category_name": "Books",
        "name": "Pride and Prejudice by Jane Austen",
        "description": "A timeless romance novel exploring themes of love, class, and societal expectations.",
        "price": 15,
        "stock": 35,
        "image_url": [
            "https://tse4.mm.bing.net/th?id=OIP.Nu7KKhRxKitVEGyTg_HaBgHaLW&pid=Api&P=0&h=180"
        ]
    },
    {
        "id": "yydlr-c2q2t-otqse-ajsxj-zf6de-3ngw6-uiwjs-lssa4-qu6bt-gymed-gim",
        "store_id": "4tx5v-6j3z7-pcdxn-drc3c-kglne-vdma5-shwrv-rorei-v3t5u-ao6xh-ghe",
        "category_name": "Clothing",
        "name": "Men's Leather Wallet",
        "description": "Sleek and stylish leather wallet with ample space for cards and cash.",
        "price": 50,
        "stock": 35,
        "image_url": [
            "https://th.bing.com/th/id/OIP.uyt01zRqhJiIILULYGDrJwHaGd?rs=1&pid=ImgDetMain"
        ]
    },
    {
        "id": "zjxeq-6r2r6-sns7i-4743u-uesfq-3g5gk-jznuv-k3hju-a32qz-b5dxe-y7e",
        "store_id": "wnxuy-3zjq6-xxycz-vltok-3cuqo-r2tlx-367kr-ncvqr-qyeki-wzklw-kno",
        "category_name": "4",
        "name": "Board Game: Settlers of Catan",
        "description": "Gather resources, build settlements, and trade with other players in this strategic board game.",
        "price": 45,
        "stock": 20,
        "image_url": [
            "https://s.catch.com.au/images/product/0030/30413/5e994d28aff64272002148.jpg"
        ]
    },
    {
        "id": "zylgs-3xsym-kw3co-s34lo-hojdp-jhmjx-5dm7f-rxylm-w6msk-5zovr-xx6",
        "store_id": "4tx5v-6j3z7-pcdxn-drc3c-kglne-vdma5-shwrv-rorei-v3t5u-ao6xh-ghe",
        "category_name": "Clothing",
        "name": "Men's Slim Fit Shirt",
        "description": "Classic slim-fit shirt made from premium cotton material.",
        "price": 35,
        "stock": 63,
        "image_url": [
            "https://spy.com/wp-content/uploads/2021/06/H2H-Mens-Casual-Slim-Fit-Short-Sleeve-T-Shirts-Cotton-Blended-Soft-Lightweight-Crew-Neck.jpg?resize=714"
        ]
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

        # data = requests.get('http://bkyz2-fmaaa-aaaaa-qaaaq-cai.localhost:4943/product')

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
            # product_name = product['name']
            # product_description = product['description']
            # product_price = product['price']
            # response = f"Product: {product_name}\nDescription: {product_description}\nPrice: {product_price} ICP\n\n"
            # dispatcher.utter_message(text=response, image=product['media_urls'][0])
            # product_name = product['name']
            # product_description = product['description']
            # product_price = product['price']
            # image_url = product['media_urls'][0]  # Assuming only one image URL per product

            # # Construct the message JSON object
            # message = {
            #     "text": f"Product: {product_name}\nDescription: {product_description}\nPrice: {product_price} ICP",
            #     "image": image_url
            # }

            # Send the message

            dispatcher.utter_message(text=product['id'])

        # dispatcher.utter_message(text=response, )



        # dispatcher.utter_message(text="Here is what do you need in our store:\nhttps://buybox.com/product=\"Red Shoe\"")

        return []