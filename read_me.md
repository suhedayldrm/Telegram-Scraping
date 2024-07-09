
# 1) download_chat (Chat Information Collector)

This project is designed to collect and analyze chat information using the Telegram API. It retrieves detailed information about chats, including member counts, message counts, and the date of the first message, while ensuring certain conditions are met before storing the data.

## Prerequisites

- Python 3.x
- `pandas` library
- `pyrogram` library
- `nest_asyncio` library

## Files

- `chat_info.csv`: Stores detailed chat information.
- `checked.csv`: Stores IDs of chats that have been checked.
- `forward.csv`: Contains a list of chat IDs to be processed.

## Setup

1. Install the required libraries:
    ```sh
    pip install pandas pyrogram nest_asyncio
    ```

2. Ensure you have the necessary Telegram API credentials to use the `pyrogram` library.

## Usage

1. **Import Dependencies:**
    ```python
    from dependencies import *
    ```

2. **Load Data:**
    ```python
    chats = pd.read_csv('chat_info.csv')
    checked_chats = pd.read_csv('checked.csv')
    forward = pd.read_csv('forward.csv')['ch_id'].tolist()
    ```

3. **Run the Main Process:**
    ```python
    nest_asyncio.apply()
    results = asyncio.run(checking(forward[11455:]))
    ```

4. **Remove Duplicates:**
    ```python
    # Drop duplicates from chats
    chats = pd.read_csv('chat_info.csv')
    chats = chats.drop_duplicates(subset=['ch_id'])
    chats.to_csv('chat_info.csv', index=False)

    # Drop duplicates from checked_chats
    checked_chats = pd.read_csv('checked.csv')
    checked_chats = checked_chats.drop_duplicates(subset=['ch_id'])
    checked_chats.to_csv('checked.csv', index=False)
    ```

## Functions

- `chat_info(username, session)`: Collects chat information such as ID, username, title, type, member count, and message count.
- `first_time(username, session)`: Retrieves the date of the first message in the chat.
- `messages(id, session)`: Collects recent messages from the chat.
- `conditions(chat)`: Checks if the chat meets certain conditions (e.g., creation time and language).
- `checking(ch_ids)`: Processes a list of chat IDs, checking if they meet the conditions and updating the CSV files accordingly.

## Example

Here is a brief example of how to use the `checking` function with a subset of chat IDs:

```python
import pandas as pd
import nest_asyncio
import asyncio

from dependencies import *

# Load chat IDs
forward = pd.read_csv('forward.csv')['ch_id'].tolist()

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Run the checking process for a subset of chat IDs
results = asyncio.run(checking(forward[11455:]))
```

This example demonstrates how to load the chat IDs, apply `nest_asyncio`, and run the `checking` function asynchronously.

---

Here's a README file summarizing the functionality of the `forward.ipynb` Jupyter notebook:

---

# 2) Forwarded Message Chat ID Collector

This project is designed to collect chat IDs from forwarded messages using the Telegram API. It retrieves IDs of chats that messages have been forwarded from and stores them in a CSV file for further analysis.

## Prerequisites

- Python 3.x
- `pandas` library
- `pyrogram` library
- `nest_asyncio` library
- `uvloop` library

## Files

- `chat_info.csv`: Stores detailed chat information.
- `checked.csv`: Stores IDs of chats that have been checked.
- `forward.csv`: Contains a list of chat IDs from forwarded messages.

## Setup

1. Install the required libraries:
    ```sh
    pip install pandas pyrogram nest_asyncio uvloop
    ```

2. Ensure you have the necessary Telegram API credentials to use the `pyrogram` library.

## Usage

1. **Import Dependencies:**
    ```python
    from dependencies import *
    ```

2. **Load Data:**
    ```python
    chats = pd.read_csv('chat_info.csv')
    ids = chats['ch_id'].tolist()
    chat_usernames = chats['ch_username'].tolist()
    sessions = [ "my_account", "my_own"]

    forward = pd.read_csv("forward.csv")
    checked_chats = pd.read_csv('checked.csv')
    checked_chats_list = checked_chats['ch_id'].tolist()
    ```

3. **Forward Message Chat ID Collector:**
    ```python
    async def forwarding(username, session):
        app = Client(session)
        forward_ids = []
        async with app:
            try:
                async for message in app.search_messages(username):
                    if message.forward_from_chat is not None:
                        forward_ids.append(message.forward_from_chat.id)
                    else:
                        continue
                return list(set(forward_ids))
            except Exception as e:
                print(f"Error getting chat dates: {e}")
                return None
    ```

4. **Run the Main Process:**
    ```python
    for chat_id in ids[207:]:
        nest_asyncio.apply()
        uvloop.install()
        index = ids.index(chat_id)
        print(index)
        forward_ids = asyncio.run(forwarding(chat_id, "my_account"))
        print(forward_ids)
        if forward_ids is not None:
            forward_ids = list(set(forward_ids))
            forward_ids = [id for id in forward_ids if id not in checked_chats_list]
            # Put them in forward.csv with the column name ["ch_id"]
            forward_df = {"ch_id": forward_ids}
            forward_df = pd.DataFrame(forward_df)
            forward_df.to_csv("forward.csv", index=False, mode="a", header=False)
        else:
            continue
    ```

5. **Remove Duplicates:**
    ```python
    forward = pd.read_csv("forward.csv")
    forward = forward.drop_duplicates()
    forward.to_csv("forward.csv", index=False)
    ```

## Functions

- `forwarding(username, session)`: Collects IDs of chats from which messages have been forwarded.
- The main process iterates over a list of chat IDs, retrieves forwarded message chat IDs, and updates the `forward.csv` file with unique IDs.

## Example

Here is a brief example of how to use the `forwarding` function within the main process:

```python
import pandas as pd
import nest_asyncio
import uvloop
import asyncio

from dependencies import *

# Load chat IDs and other data
chats = pd.read_csv('chat_info.csv')
ids = chats['ch_id'].tolist()
checked_chats = pd.read_csv('checked.csv')
checked_chats_list = checked_chats['ch_id'].tolist()

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Iterate and process each chat ID
for chat_id in ids[207:]:
    uvloop.install()
    index = ids.index(chat_id)
    print(index)
    forward_ids = asyncio.run(forwarding(chat_id, "my_account"))
    print(forward_ids)
    if forward_ids is not None:
        forward_ids = list(set(forward_ids))
        forward_ids = [id for id in forward_ids if id not in checked_chats_list]
        forward_df = {"ch_id": forward_ids}
        forward_df = pd.DataFrame(forward_df)
        forward_df.to_csv("forward.csv", index=False, mode="a", header=False)
    else:
        continue

# Remove duplicates from the forward.csv file
forward = pd.read_csv("forward.csv")
forward = forward.drop_duplicates()
forward.to_csv("forward.csv", index=False)
```

This example demonstrates how to load the chat IDs, apply `nest_asyncio`, and run the main process to collect forwarded message chat IDs asynchronously.

---

Here's a README file summarizing the functionality of the `messages.ipynb` Jupyter notebook:

---

# 3) Chat Messages Collector

This project is designed to collect messages from specific chats using the Telegram API. It retrieves messages from chats, filtering by a specific date threshold, and saves the messages in separate CSV files for each chat.

## Prerequisites

- Python 3.x
- `pandas` library
- `pyrogram` library
- `nest_asyncio` library

## Files

- `chats/`: Directory where individual chat message CSV files are saved.

## Setup

1. Install the required libraries:
    ```sh
    pip install pandas pyrogram nest_asyncio
    ```

2. Ensure you have the necessary Telegram API credentials to use the `pyrogram` library.

## Usage

1. **Import Dependencies:**
    ```python

    from dependencies import *
    ```

2. **Define the Asynchronous Function:**
    ```python
    from datetime import datetime

    async def achat(chat_id):
        app = Client("my_account")
        messages = []
        async with app:
            async for message in app.search_messages(chat_id):
                threshold = datetime.strptime('2017.12.30', '%Y.%m.%d')
                if message.date and message.text is not None:
                    message_date = message.date
                    if message_date < threshold:
                        message = {
                            "date": message_date.strftime("%Y.%m.%d"),
                            "message_text": message.text
                        }
                        messages.append(message)
        return messages
    ```

3. **Run the Function:**
    ```python
    from dependencies import *
    from pyrogram import Client
    import asyncio
    asyncio.run(achat("achat"))
    ```

4. **List of Chat IDs:**
    ```python
    ids = [-1001106299859]
    ```

5. **Save Messages to CSV:**
    ```python
    from dependencies import *
    
    for chat_id in ids:
        nest_asyncio.apply()
        messages = asyncio.run(achat(chat_id))
        # Index printing
        print(ids.index(chat_id), chat_id)
        if messages is not None:
            df = pd.DataFrame(messages)
            df.to_csv(f"chats/{chat_id}.csv", index=False)
        else:
            print(f"messages: messages are empty for {chat_id}")
    ```

## Functions

- `achat(chat_id)`: Collects messages from a chat, filtering messages by a date threshold, and returns a list of messages.

## Example

Here is a brief example of how to use the `achat` function and save messages to CSV files:

```python
import sys
import os
import datetime
import pandas as pd
from dependencies import *
from pyrogram import Client
import nest_asyncio
import asyncio

# Define asynchronous function
async def achat(chat_id):
    app = Client("my_account")
    messages = []
    async with app:
        async for message in app.search_messages(chat_id):
            threshold = datetime.strptime('2017.12.30', '%Y.%m.%d')
            if message.date and message.text is not None:
                message_date = message.date
                if message_date < threshold:
                    message = {
                        "date": message_date.strftime("%Y.%m.%d"),
                        "message_text": message.text
                    }
                    messages.append(message)
    return messages

# List of chat IDs
ids = [-1001106299859]

# Save messages to CSV files
for chat_id in ids:
    nest_asyncio.apply()
    messages = asyncio.run(achat(chat_id))
    # Index printing
    print(ids.index(chat_id), chat_id)
    if messages is not None:
        df = pd.DataFrame(messages)
        df.to_csv(f"chats/{chat_id}.csv", index=False)
    else:
        print(f"messages: messages are empty for {chat_id}")
```

This example demonstrates how to set up the environment, define the asynchronous function to collect messages, and save the messages to CSV files for each chat.

---

Here's the README file with a shorter example:

---

# 4) Topic Modeling for Chat Messages

This project is designed to perform topic modeling on chat messages using the Latent Dirichlet Allocation (LDA) algorithm. It processes chat messages, cleans the text data, tokenizes and lemmatizes the words, and then identifies topics within the chat data.

## Prerequisites

- Python 3.x
- `pandas` library
- `nltk` library
- `spacy` library
- `gensim` library
- `wordcloud` library
- `matplotlib` library

## Setup

1. Install the required libraries:
    ```sh
    pip install pandas nltk spacy gensim wordcloud matplotlib
    ```

2. Download the required NLTK data:
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    ```

3. Download the SpaCy language model:
    ```sh
    python -m spacy download en_core_web_sm
    ```

## Usage

1. **Import Dependencies:**
    ```python
    from dependencies import *
    import pandas as pd
    import re
    import emoji
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    import spacy
    from gensim import corpora, models
    import pickle
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    ```

2. **Load and Clean Data:**
    ```python
    df = pd.read_csv('-1001323856649.csv')
    messages = df["translated"].tolist()

    cleaned_list = []
    for x in messages:
        x = re.split('https://.*', str(x))[0]  # remove links
        x = ' '.join(re.sub("[0-9.,!?:;\\-='\"@#_]", " ", emoji.demojize(x)).split())  # remove punctuation, numbers, and emojis
        if x != 'nan':  # check if x is not nan
            cleaned_list.append(x.lower())  # lowercase and append to cleaned_list
    cleaned_list = [x for x in cleaned_list if x not in [None, 'none', '', 'nan']]  # remove unwanted entries
    ```

3. **Tokenize and Remove Stopwords:**
    ```python
    tokenized_list = [word_tokenize(x) for x in cleaned_list]
    tokenized_list = [x for x in tokenized_list if len(x) > 1]  # remove single characters

    stop_words = set(stopwords.words('english'))
    for i in range(len(tokenized_list)):
        tokenized_list[i] = [x for x in tokenized_list[i] if x not in stop_words]  # remove stopwords
        tokenized_list[i] = [x for x in tokenized_list[i] if len(x) > 1]  # remove single characters
        tokenized_list[i] = [x for x in tokenized_list[i] if x.isalpha()]  # remove non-alphabetic characters
    tokenized_list = [x for x in tokenized_list if len(x) > 1]  # remove single characters
    ```

4. **Lemmatize Tokens:**
    ```python
    nlp = spacy.load("en_core_web_sm")
    lemmatized_list = [[x for x in tokenized_list[i] if len(x) > 1] for i in range(len(tokenized_list))]  # remove single characters
    ```

5. **Create LDA Model:**
    ```python
    dictionary = corpora.Dictionary(lemmatized_list)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in lemmatized_list]

    lda_model = models.LdaModel(
        corpus=doc_term_matrix,
        id2word=dictionary,
        num_topics=20,
        passes=20
    )

    topics = lda_model.print_topics(num_words=40)
    for topic in topics:
        words = topic[1].split(" + ")
        print("Topic {}: {}".format(topic[0], ", ".join(word.split("*")[1] for word in words)))

    with open('lda_model.pkl', 'wb') as f:
        pickle.dump(lda_model, f)
    ```

6. **Visualize Topics:**
    ```python
    topic_to_visualize = 19
    with open('lda_model.pkl', 'rb') as f:
        lda_model = pickle.load(f)

    words = lda_model.show_topic(topic_to_visualize, 30)
    word_dict = {word: prob for word, prob in words}
    wordcloud = WordCloud(width=800, height=400, random_state=21, max_font_size=110, background_color='white').generate_from_frequencies(word_dict)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.savefig(f'figure_{topic_to_visualize}.png')
    plt.show()
    ```



---


Here's a README file summarizing the functionality of the `relevance.ipynb` Jupyter notebook:

---

# 5) Chat Relevance Scorer

This project is designed to score the relevance of chat messages based on the presence of certain keywords. It processes chat messages, calculates relevance scores, and stores the results in a CSV file.

## Prerequisites

- Python 3.x
- `pandas` library
- `pyrogram` library
- `nest_asyncio` library

## Setup

1. Install the required libraries:
    ```sh
    pip install pandas pyrogram nest_asyncio
    ```

2. Ensure you have the necessary Telegram API credentials to use the `pyrogram` library.

## Usage

1. **Import Dependencies:**
    ```python

    from dependencies import *
    import pandas as pd
    import asyncio
    import nest_asyncio
    ```

2. **Define Keyword List:**
    ```python
    keyword_list = []
    ```

3. **Define Relevance Scoring Function:**
    ```python
    async def relevance_score(chat_id, keyword_list, threshold=0.1):
        sessions = ["my_account", "p_account"]
        tasks = [messages(chat_id, session) for session in sessions]
        messages_info_list = await asyncio.gather(*tasks)
        
        if all([item is None for item in messages_info_list]):
            return None
        
        messages_info = [item for sublist in messages_info_list if sublist is not None for item in sublist]
        messages_info = list(set(messages_info))
        
        message_count = len(messages_info)
        found_keywords = set()
        found = 0
        
        for message in messages_info:
            found_keywords.update([keyword for keyword in keyword_list if message.lower().count(keyword) > 0])
            found += sum([message.lower().count(keyword) for keyword in keyword_list])
        
        keywords_count = len(found_keywords)
        message_count = len(messages_info)
        
        relevance_score = (found / message_count) * (1 + 0.5 * (keywords_count / len(keyword_list)))
        raw_score = keywords_count / len(keyword_list)

        if raw_score <= threshold:
            print("Not that much Relevant")
            relevance_score -= 1
        else:
            print("Highly Relevant")
        
        result = {
            "ch_id": chat_id,
            "relevance_score": relevance_score,
            "raw_score": raw_score,
            "keywords_count": keywords_count,
            "found_keywords": list(found_keywords)
        }
        return result
    ```

4. **Run Relevance Scoring Process:**
    ```python
    relevant_chats = pd.read_csv('relevant.csv')
    chats = pd.read_csv('chat_info.csv')
    chat_ids = chats['ch_id'].tolist()

    nest_asyncio.apply()
    
    for chat_id in chat_ids:
        result = asyncio.run(relevance_score(chat_id, keyword_list))
        if result is None:
            continue
        
        result_df = pd.DataFrame([result])
        result_df.to_csv('relevant.csv', mode='a', header=False, index=False)
    ```

5. **Remove Duplicates:**
    ```python
    relevant_chats = pd.read_csv('relevant.csv')
    relevant_chats = relevant_chats.drop_duplicates()
    relevant_chats.to_csv('relevant.csv', index=False)
    ```

## Functions

- `relevance_score(chat_id, keyword_list, threshold)`: Calculates the relevance score for a chat based on the presence of specified keywords.


```

This example demonstrates how to set up the environment, define the relevance scoring function, and run the process to calculate and store relevance scores for chat messages.

---