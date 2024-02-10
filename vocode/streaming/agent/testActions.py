# import json
# import asyncio
# from typing import Optional, List, Dict, Any
# from openai import AsyncClient,Client
# import logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)
# base_url = "https://c50f5b7060f7e7fe.ngrok.app/v1"
# aclient = AsyncClient(api_key="uindciun", base_url=base_url)
# # list of dictionaries of role, content

# #expected: true
# transferTest = [
#   {
#     "role": "assistant",
#     "content": "Hello, this is Property Care Management's OpenCall assistant. How may I help you today?"
#   },
#   {
#     "role": "user",
#     "content": "hello this is brad i'm calling because i'm a renter with anne and i need to talk with her because i've got a pipe burst"
#   },
#   {
#     "role": "assistant",
#     "content": "I will immediately transfer your call to Anne or the appropriate response team to address this urgent issue. Please hold while I connect you."
#   }
# ]

# tests = [
#     [
#         [
#             {"role": "assistant", "content": "Hello, this is OpenCall Anne Silverstein's assistant. How may I assist you today?"},
#             {"role": "user", "content": "Hello, this is the plumber Anne called about the burst pipe. Could you let her know I've fixed the pipe and will be sending an invoice for my services."},
#             {"role": "assistant", "content": "I can pass your message along. Thanks!"},
#         ],
#         True,
#         "send_message"
#     ],
#     [
#         [
#             {"role": "assistant", "content": "Hello, this is OpenCall Anne Silverstein's assistant. How may I assist you today?"},
#             {"role": "user", "content": "Hello, this is the plumber Anne called about the burst pipe. Could you pass a message along?"},
#             {"role": "assistant", "content": "Sure! What are the details I should include?"},
#         ],
#         False,
#         "none"
#     ],
#     [
#         [
#             {"role": "assistant", "content": "Hello, this is OpenCall Anne Silverstein's assistant. How may I assist you today?"},
#             {"role": "user", "content": "Hello, this is the plumber Anne called about the burst pipe. I need to speak with her about the water damage"},
#             {"role": "assistant", "content": "I'll transfer you now!"},
#         ],
#         True,
#         "transfer_call"
#     ],
#     [
#         [
#             {"role": "assistant", "content": "Hello, this is OpenCall Anne Silverstein's assistant. How may I assist you today?"},
#             {"role": "user", "content": "I need to speak with Anne about something. Could you please connect me."},
#             {"role": "assistant", "content": "Before I transfer you I need to understand the purpose of your call."},
#         ],
#         False,
#         "none"
#     ],
#     [
#         [
#             {"role": "assistant", "content": "Hello, this is OpenCall Anne Silverstein's assistant. How may I assist you today?"},
#             {"role": "user", "content": "I'm just calling to prank your business."},
#             {"role": "assistant", "content": "I'm sorry, but I must terminate this call if it's not business-related. Please call us back for any legitimate inquiries."},
#         ],
#         True,
#         "hangup_call"
#     ],
#     [
#         [
#             {"role": "assistant", "content": "Hello, this is OpenCall Anne Silverstein's assistant. How may I assist you today?"},
#             {"role": "user", "content": "Hi is Anne there?"},
#             {"role": "assistant", "content": "Anne is busy, can I take a message?"},
#         ],
#         True,
#         "none"
#     ],
#     [
#         [
#             {"role": "assistant", "content": "Hello, this is OpenCall Anne Silverstein's assistant. How may I assist you today?"},
#             {"role": "user", "content": "Hi , I'd like to be transferred to Anne."},
#             {"role": "assistant", "content": "Hello, may I ask the purpose of your call?"},
#         ],
#         True,
#         "none"
#     ],
#     [
#         [
#             {"role": "assistant", "content": "Hello, this is OpenCall Anne Silverstein's assistant. How may I assist you today?"},
#             {"role": "user", "content": "Hi , I'd like to be transferred to Anne."},
#             {"role": "assistant", "content": "Hello, may I ask the purpose of your call?"},
#             {"role": "user", "content": "I need to speak with her about the rent next month."},
#             {"role": "assistant", "content": "What is your name and number? I will pass the message along."},
#         ],
#         True,
#         "none"
#     ],
#     [
#         [
#             {"role": "assistant", "content": "Hello, this is OpenCall Anne Silverstein's assistant. How may I assist you today?"},
#             {"role": "user", "content": "Hi , I'd like to be transferred to Anne."},
#             {"role": "assistant", "content": "Hello, may I ask the purpose of your call?"},
#             {"role": "user", "content": "I need to speak with her about the rent next month."},
#             {"role": "assistant", "content": "What is your name and number? I will pass the message along."},
#             {"role": "user", "content": "My name is John and my number is 123-456-7890."},
#             {"role": "assistant", "content": "Thank you, I will notify her."},
#         ],
#         True,
#         "send_message"
#     ],
#     [
#         [
#             {"role": "assistant", "content": "Hello, this is OpenCall Anne Silverstein's assistant. How may I assist you today?"},
#             {"role": "user", "content": "I'm dissatisfied with your service."},
#             {"role": "assistant", "content": "I apologize for any inconvenience. Can you explain the issue so we can try to resolve it?"},
#         ],
#         False,
#         "none"
# ],
# ]
# tools =[
#         {
#             "type": "function",
#             "function": {
#                 "name": "send_message",
#                 "description": "Pass a message/ notification to the receiver.",
#                 "parameters": {
#                     "type": "object",
#                     "properties": {
#                         "message_to_pass": {
#                             "type": "string",
#                             "description": "The message to pass, limited to 300 characters"
#                         }
#                     },
#                     "required": ["message_to_pass"]
#                 }
#             }
#         },
#         {
#             "type": "function",
#             "function": {
#                 "name": "transfer_call",
#                 "description": "Transfer the call with a reason",
#                 "parameters": {
#                     "type": "object",
#                     "properties": {
#                         "transfer_reason": {
#                             "type": "string",
#                             "description": "The reason for transferring the call, limited to 120 characters"
#                         }
#                     },
#                     "required": ["transfer_reason"]
#                 }
#             }
#         },
#         {
#             "type": "function",
#             "function": {
#                 "name": "hangup_call",
#                 "description": "End the call with a reason",
#                 "parameters": {
#                     "type": "object",
#                     "properties": {
#                         "end_reason": {
#                             "type": "string",
#                             "description": "The reason for ending the call, limited to 120 characters"
#                         }
#                     },
#                     "required": ["end_reason"]
#                 }
#             }
#         }
#     ]
# def get_chat_parameters(
#             messages: Optional[List] = None, use_functions: bool = True
#     ):
#         messages = messages
#         parameters: Dict[str, Any] = {
#             "messages": messages,
#             "max_tokens": 400,
#             "temperature": 0,
#             "stop": ["User:", "<|im_end|>", "?"],
#         }
#         parameters["model"] = "TheBloke/Nous-Hermes-2-Mixtral-8x7B-DPO-AWQ"
#         return parameters
# async def process_action(test):
#     try:
#         messages = test[0]
#         expected = test[1]
#         tool_used = test[2]
#         logger.info(f"Processing action classification...")
#         # Get available tool names and their descriptions
#         tool_descriptions = [f"'{tool['function']['name']}': {tool['function']['description']} (Required params: {', '.join(tool['function']['parameters']['required'])})" for tool in tools]
#         pretty_tool_descriptions = ', '.join(tool_descriptions)
#         logger.info(f"Processing action classification...")
#         stringified_messages = json.dumps(messages)
#         logger.debug(f"Stringified messages: {stringified_messages}")
#         # Construct the prompt to guide the model to generate a valid JSON
#         preamble = f"""You will be provided with a conversation transcript between a caller and the receiver's assistant. Your task is to determine whether the receiver's assistant, in their latest message, informed the caller they would do an action, after having collected any requirement information from the caller. The possible actions that can be done (either during or after the call) are: {pretty_tool_descriptions}.\nIf the receiver's assistant promised they would do an action and they have the've already collected any information required to do that action, return the action name from the list provided. Otherwise, if the receiver's assistant is still collecting required action inputs from the caller, they can't do the action yet so you must return 'None'. Return a single word."""
#         system_message = {"role": "system", "content": preamble}
#         transcript_message = {"role": "user", "content": stringified_messages}
#         combined_messages = [system_message, transcript_message]
#         chat_parameters = get_chat_parameters(messages=combined_messages)
#         response = await aclient.chat.completions.create(**chat_parameters)
#         tool_classification = response.choices[0].message.content.lower().strip()
#         if tool_used not in tool_classification:
#             logger.info(f"Test failed. Got: {tool_classification}\nMessages: {messages}")
#             return
#     except Exception as e:
#         logger.error(f"An error occurred: {e}")
# async def main():
#     #modify to call all the tests in parallel (iterating through tests) and print the results
#     tasks = [process_action(test) for test in tests]
#     await asyncio.gather(*tasks)

# if __name__ == "__main__":
#     asyncio.run(main())


import json
import asyncio
from typing import Optional, List, Dict, Any
from openai import AsyncClient,Client
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
base_url = "https://c50f5b7060f7e7fe.ngrok.app/v1"
aclient = AsyncClient(api_key="uindciun", base_url=base_url)
# list of dictionaries of role, content


tests = [
    [
        [
            {"role": "assistant", "content": "Hello, this is OpenCall Anne Silverstein's assistant. How may I assist you today?"},
            {"role": "user", "content": "Hello, this is the plumber Anne called about the burst pipe. Could you let her know I've fixed the pipe and will be sending an invoice for my services."},
            {"role": "assistant", "content": "I can pass your message along. Thanks!"},
        ],
        True,
        "send_message"
    ],
    [
        [
            {"role": "assistant", "content": "Hello, this is OpenCall Anne Silverstein's assistant. How may I assist you today?"},
            {"role": "user", "content": "Hello, this is the plumber Anne called about the burst pipe. Could you pass a message along?"},
            {"role": "assistant", "content": "Sure! What are the details I should include?"},
        ],
        False,
        "none"
    ],
    [
        [
            {"role": "assistant", "content": "Hello, this is OpenCall Anne Silverstein's assistant. How may I assist you today?"},
            {"role": "user", "content": "Hello, this is the plumber Anne called about the burst pipe. I need to speak with her about the water damage"},
            {"role": "assistant", "content": "I'll transfer you now!"},
        ],
        True,
        "transfer_call"
    ],
    [
        [
            {"role": "assistant", "content": "Hello, this is OpenCall Anne Silverstein's assistant. How may I assist you today?"},
            {"role": "user", "content": "I need to speak with Anne about something. Could you please connect me."},
            {"role": "assistant", "content": "Before I transfer you I need to understand the purpose of your call."},
        ],
        False,
        "none"
    ],
    [
        [
            {"role": "assistant", "content": "Hello, this is OpenCall Anne Silverstein's assistant. How may I assist you today?"},
            {"role": "user", "content": "I'm just calling to prank your business."},
            {"role": "assistant", "content": "I'm sorry, but I must terminate this call if it's not business-related. Please call us back for any legitimate inquiries. Thank you."},
        ],
        True,
        "hangup_call"
    ],
    [
        [
            {"role": "assistant", "content": "Hello, this is OpenCall Anne Silverstein's assistant. How may I assist you today?"},
            {"role": "user", "content": "Hi is Anne there?"},
            {"role": "assistant", "content": "Anne is busy, can I take a message?"},
        ],
        True,
        "none"
    ],
    [
        [
            {"role": "assistant", "content": "Hello, this is OpenCall Anne Silverstein's assistant. How may I assist you today?"},
            {"role": "user", "content": "Hi , I'd like to be transferred to Anne."},
            {"role": "assistant", "content": "Hello, may I ask the purpose of your call?"},
        ],
        True,
        "none"
    ],
    [
        [
            {"role": "assistant", "content": "Hello, this is OpenCall Anne Silverstein's assistant. How may I assist you today?"},
            {"role": "user", "content": "Hi , I'd like to be transferred to Anne."},
            {"role": "assistant", "content": "Hello, may I ask the purpose of your call?"},
            {"role": "user", "content": "I need to speak with her about the rent next month."},
            {"role": "assistant", "content": "What is your name and number? I will pass the message along."},
        ],
        True,
        "none"
    ],
    [
        [
            {"role": "assistant", "content": "Hello, this is OpenCall Anne Silverstein's assistant. How may I assist you today?"},
            {"role": "user", "content": "Hi , I'd like to be transferred to Anne."},
            {"role": "assistant", "content": "Hello, may I ask the purpose of your call?"},
            {"role": "user", "content": "I need to speak with her about the rent next month."},
            {"role": "assistant", "content": "What is your name and number? I will pass the message along."},
            {"role": "user", "content": "My name is John and my number is 123-456-7890."},
            {"role": "assistant", "content": "Thank you, I will notify her."},
        ],
        True,
        "send_message"
    ],
    [
        [
            {"role": "assistant", "content": "Hello, this is OpenCall Anne Silverstein's assistant. How may I assist you today?"},
            {"role": "user", "content": "I'm dissatisfied with your service."},
            {"role": "assistant", "content": "I apologize for any inconvenience. Can you explain the issue so we can try to resolve it?"},
        ],
        False,
        "none"
    ],
    [
        [
            {"role": "assistant", "content": "Hello, this is OpenCall Anne Silverstein's assistant. How may I assist you today?"},
            {"role": "user", "content": "I'm calling to reach Anne about her car's warranty."},
            {"role": "assistant", "content": "I apologize for any inconvenience but we can't take this call. Please remove this number from your call list. Bye."},
        ],
        False,
        "hangup_call"
    ],
    [
        [
            {"role": "assistant", "content": "Hello, this is OpenCall Anne Silverstein's assistant. How may I assist you today?"},
            {"role": "user", "content": "I'm calling to reach Anne about her car's warranty."},
            {"role": "assistant", "content": "I apologize for any inconvenience but we can't take this call. Please remove this number from your call list. Bye."},
        ],
        False,
        "hangup_call"
    ],
    [
        [
            {"role": "assistant", "content": "Hello, this is OpenCall Anne Silverstein's assistant. How may I assist you today?"},
            {"role": "user", "content": "I'm a scammer from Nigeria. I wanted to discuss a business opportunity with Anne."},
            {"role": "assistant", "content": "I'm sorry, but I must terminate this call if it's not business-related. Please call us back for any legitimate inquiries."},
        ],
        False,
        "hangup_call"
    ],
]
tools =[
        {
            "type": "function",
            "function": {
                "name": "send_message",
                "description": "Send a message/ notification to the receiver.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message_to_pass": {
                            "type": "string",
                            "description": "The message to pass, limited to 300 characters"
                        }
                    },
                    "required": ["message_to_pass"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "transfer_call",
                "description": "Transfer the call with a reason",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "transfer_reason": {
                            "type": "string",
                            "description": "The reason for transferring the call, limited to 120 characters"
                        }
                    },
                    "required": ["transfer_reason"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "hangup_call",
                "description": "Hangup the call with a reason",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "end_reason": {
                            "type": "string",
                            "description": "The reason for ending the call, limited to 120 characters"
                        }
                    },
                    "required": ["end_reason"]
                }
            }
        }
    ]
def get_chat_parameters(
            messages: Optional[List] = None, use_functions: bool = True
    ):
        messages = messages
        parameters: Dict[str, Any] = {
            "messages": messages,
            "max_tokens": 400,
            "temperature": 0,
            "stop": ["User:", "<|im_end|>", "?"],
        }
        parameters["model"] = "TheBloke/Nous-Hermes-2-Mixtral-8x7B-DPO-AWQ"
        return parameters
async def process_action(test):
    try:
        messages = test[0]
        expected = test[1]
        tool_used = test[2]
        logger.info(f"Processing action classification...")
        # Get available tool names and their descriptions
        tool_descriptions = [f"'{tool['function']['name']}': {tool['function']['description']} (Required params: {', '.join(tool['function']['parameters']['required'])})" for tool in tools]
        pretty_tool_descriptions = ', '.join(tool_descriptions)
        logger.info(f"Processing action classification...")
        stringified_messages = str(messages)
        logger.debug(f"Stringified messages: {stringified_messages}")
        # Construct the prompt to guide the model to generate a valid JSON
        preamble = f"""You will be provided with a conversational transcript between a caller and the receiver's assistant. During the conversation, the assistant has the following actions it can take: {pretty_tool_descriptions}.\n Your task is to infer whether, currently, the assistant is waiting for the caller to respond, or is immediately going to execute an action without waiting for a response. Return the action name from the list provided if the assistant is executing an action. If the assistant is waiting for a response, return 'None'. Return a single word."""
        system_message = {"role": "system", "content": preamble}
        transcript_message = {"role": "user", "content": stringified_messages}
        combined_messages = [system_message, transcript_message]
        chat_parameters = get_chat_parameters(messages=combined_messages)
        response = await aclient.chat.completions.create(**chat_parameters)
        tool_classification = response.choices[0].message.content.lower().strip()
        if tool_used not in tool_classification:
            logger.info(f"Test failed. Got: {tool_classification}\nMessages: {messages}")
            return
    except Exception as e:
        logger.error(f"An error occurred: {e}")
async def main():
    #modify to call all the tests in parallel (iterating through tests) and print the results
    tasks = [process_action(test) for test in tests]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())


