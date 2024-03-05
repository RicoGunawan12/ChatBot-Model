# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
#
#
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
        print(tracker.latest_message.get('text'))

        dispatcher.utter_message(text="Here is what do you need in our store:\nhttps://buybox.com/product=\"Red Shoe\"")

        return []