import requests
import json
import re
from rapidfuzz import process, fuzz
import nltk
from nltk.corpus import words
import time
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# 🔹 Скачать словарь английских слов при первом запуске
nltk.download("words", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

english_words = set(words.words())

# 🔹 Словарь сленга (можно пополнять)
slang_words = {
    # базовый крипто-сленг
    "gm", "gn", "rekt", "pamp", "dump", "moon", "lambo", "hodl", "fomo", "jomo",
    "ngmi", "wagmi", "btd", "btdi", "btfd", "ape", "apestrong", "shill", "rug",
    "rugged", "rugpull", "bagholder", "bags", "degen", "paperhands", "diamondhands",
    "whale", "shrimp", "pleb", "sat", "sats", "stacking", "stackingsats", "alpha",
    "beta", "gigachad", "plebs", "autist", "autistic", "ser", "frens", "wagie",
    "anon", "broski", "chad", "giga", "apeing", "apes", "cope", "copeium", "copium",
    "hopium", "moonbois", "moonboy", "moonboys", "lasereyes", 'virtual', 'memecoins', 'normie', 'gme',
    'run', 'crypto'

    # тикеры и аббревиатуры
     "wojak", "memecoin", "alts", "alt",

    # торговые выражения
    "long", "short", "pump", "dump", "flippening", "alltimehigh", "ath", "dip",
    "buythedip", "sellthetop", "moonshot", "parabolic", "gigapump", "gigadump",
    "bottom", "top", "bullrun", "bearrun", "capitulation", "deadcat", "doubletop",
    "doublebottom", "rangebound", "sideways", "exitliquidity", "bagging",

    # NFT / Web3
    "floor", "mint", "minted", "airdrop", "drop", "gmers", "degenmint", "delist",
    "delisted", "flooring", "sweep", "sweeping", "wagmiarmy", "metaverse",
    "metaversebro", "pfp", "pfps", "jpeg", "jpegs", "rightclicksave", "opensea",
    "blur", "rarible", "magiceden", "onchain", "offchain", "dao", "defi",
    "shitcoin", "shitcoins", "shitcoiner", "tokenomics", "ponzinomics",

    # эмоции и мемы
    "lol", "lmao", "rofl", "kek", "xd", "vibe", "sus", "yeet", "bruh", "pog",
    "poggers", "smh", "idk", "irl", "afk", "imo", "imho", "geez", "noob", "ezpz",
    "wheee", "stonks", "stonk", "tendies", "yolo", "apein", "apeout", "fud",
    "fudding", "based", "cringe", "vibin", "glhf", "pwned", "owned", "clown",
    "clownworld", "sadge", "monkaS", "peepo", "omegalul", "lulw", "lul", "5head",
    "galaxybrain", "npc", "seethe", "kekw", "xdd", "gigachad", "soyboy", "simp",
    "doomer", "bloomer", "coomer", "boomer", "zoomer", "sigma", "betaorbiter",
    "topg", "gyatt",

    # прочее сетевое
    "np", "idc", "ffs", "omg", "wtf", "gg", "ez", "wp", "lfg", "goat", "mfers",
    "toobased", "vitalik", "satoshi", "elon", "cz", "bro", "bros", "fam", "irlbro",
    "basedgod", "shitpost", "shitposter", "shitposting", "okboomer", "okzoomers",
    "degens", "gigabrain", "megalul", "bigbrain", "smartmoney", "newbie", "pajeet",
    "wagmiarmy", "anonarmy", "pumpndump", "insiders", "apestrong", "hodler",
    "cryptobros", "cryptobro", "moonmission", "diamondhanded", "paperhanded", "nft"
}

extended_slang = slang_words | {w + "s" for w in slang_words}

def is_english_word(word: str) -> bool:
    """
    Проверяет, является ли слово английским:
    1) Есть ли оно в nltk.corpus.words
    2) Есть ли для него хоть один synset в WordNet
    """
    if word in english_words:
        return True
    if wordnet.synsets(word):  # есть ли синонимы в wordnet
        return True
    return False

def fetch_top_coins_with_pairs(output_path="tickers_pairs.json",
                               min_market_cap=50000000,
                               min_volume=1000,
                               max_pages=8):
    """
    Загружает монеты с CoinGecko, фильтрует по капитализации и объему торгов,
    и сохраняет пары тикер -> полное имя в JSON.
    С логами: страницы, количество токенов, ожидание при лимите API.
    """
    import time, requests, json, re

    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {"vs_currency": "usd", "order": "market_cap_desc", "per_page": 250, "page": 1}

    ticker_name_pairs = dict()
    pattern = re.compile(r"^[a-z0-9]+$")

    while params["page"] <= max_pages:
        print(f"🔹 Обрабатываем страницу {params['page']}...")
        for attempt in range(5):
            try:
                response = requests.get(url, params=params)
                if response.status_code == 429:  # Too Many Requests
                    wait_time = 10 * (attempt + 1)
                    print(f"⚠️ Лимит API, ждём {wait_time} сек...")
                    time.sleep(wait_time)
                    continue
                response.raise_for_status()
                break
            except requests.exceptions.HTTPError as e:
                wait_time = 5
                print(f"❌ Ошибка {e}, повтор через {wait_time} сек...")
                time.sleep(wait_time)
        else:
            print("❌ Не удалось получить данные, выходим")
            break

        data = response.json()
        if not data:
            print("⚠️ Пустая страница, выходим")
            break

        new_pairs = 0
        for coin in data:
            market_cap = coin.get("market_cap") or 0
            volume = coin.get("total_volume") or 0
            if market_cap >= min_market_cap and volume >= min_volume:
                name = coin["name"].lower().replace(" ", "")
                symbol = coin["symbol"].lower()
                if 2 <= len(symbol) <= 10 and pattern.match(symbol):
                    if symbol not in ticker_name_pairs:
                        ticker_name_pairs[symbol] = name
                        new_pairs += 1

        print(f"✅ Страница {params['page']} обработана, добавлено {new_pairs} новых пар, всего {len(ticker_name_pairs)}")
        params["page"] += 1
        time.sleep(1)  # пауза, чтобы не словить 429

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ticker_name_pairs, f, ensure_ascii=False, indent=2)

    print(f"🎉 Сохранено {len(ticker_name_pairs)} пар тикер->имя в {output_path}")


lemmatizer = WordNetLemmatizer()

def find_crypto_mentions_v2(text, ticker_name_pairs, threshold=85):
    text = text.lower()
    words_in_text = re.findall(r"[a-zA-Z0-9$]+", text)

    matches = set()
    tickers = list(ticker_name_pairs.keys())
    names = list(ticker_name_pairs.values())

    for word in words_in_text:
        if len(word) < 2:
            continue

        lemma = lemmatizer.lemmatize(word, wordnet.NOUN)
        lemma = lemmatizer.lemmatize(lemma, wordnet.VERB)

        if is_english_word(lemma):
            continue
        if word in extended_slang or lemma in extended_slang:
            continue

        # 🔹 точное совпадение с тикером
        if word in tickers:
            matches.add(word)
            continue

        # 🔹 точное совпадение с именем
        if word in names:
            # добавляем только тикер
            ticker = [t for t, n in ticker_name_pairs.items() if n == word][0]
            matches.add(ticker)
            continue

        # 🔹 fuzzy тикер
        result = process.extractOne(word, tickers, scorer=fuzz.ratio, score_cutoff=threshold)
        if result:
            candidate, score, _ = result
            if score >= 83:
                matches.add(candidate)
                continue

        # 🔹 fuzzy имя
        result = process.extractOne(word, names, scorer=fuzz.ratio, score_cutoff=threshold)
        if result:
            candidate_name, score, _ = result
            if score >= 88 and (len(word) >= 4 or score > 92):
                ticker = [t for t, n in ticker_name_pairs.items() if n == candidate_name][0]
                matches.add(ticker)

    matches = {m for m in matches if m not in slang_words}
    return sorted(matches)

import re

def clean_text(text):
    def process_word(word):
        # удаляем слова длиной 1
        if len(word) == 1:
            return ''
        # удаляем слова, содержащие хотя бы одну цифру
        if any(char.isdigit() for char in word):
            return ''
        # если слово длиной >=3 и оканчивается на z или s, убираем последнюю букву
        if len(word) >= 3 and word.lower()[-1] in ('z', 's'):
            return word[:-1]
        return word

    # обрабатываем каждое слово через регулярку
    return re.sub(r'\b\w+\b', lambda m: process_word(m.group()), text)



# --- Пример работы ---
#fetch_top_coins_with_pairs("clean_names_and_symbols.json")

with open("clean_names_and_symbols.json", "r", encoding="utf-8") as f:
    tickers = json.load(f)

text = """"Post 1:
🚀🚀 $BTC TO THE MOON! JUST BROKE 69K AGAIN! WHALES ARE LOADING UP, DON'T MISS OUT! THIS IS NOT FINANCIAL ADVICE BUT LFG!!! 💎🙌
Comments:

"Bought the dip at 68.5k, we going to 100k EOY ez"

"FOMOing in rn, sold my kidney for this"

"RIP bears 😂"

"IM NOT FUCKIN SELLING"

Post 2:
$ETH GAS FEES ARE KILLING ME RN 😭 JUST PAID $200 FOR A SWAP. SERIOUSLY, WEN L2 ADOPTION? ARBITRUM AND POLYGON SAVING MY ASS BUT STILL...

Comments:

"Try $SOL, fees are literally pennies"

"Layer 2 is the future, change my mind"

"This is why I stick to CEXs lol"

"Just use BSC bro, who cares about decentralization"

Post 3:
Y'ALL SLEEPING ON $DOGE AGAIN? ELON TWEETED "WOOF" AND IT PUMPED 20%. I KNOW IT'S A MEME BUT THE PEOPLE'S COIN ALWAYS WINS. #DOGETO1DOLLAR

Comments:

"Doge is a shitcoin, change my mind"

"Bought 10k coins at 0.05, still holding"

"Elon manipulates the market and y'all eat it up"

"Remember when it hit 0.70? Good times"

Post 4:
RUG PULL ALERT 🚨 $SAFEMOON 2.0 DEV WALLET DUMPED 10M TOKENS. PRICE DOWN 80% IN 1 HOUR. IF YOU'RE STILL IN THESE SHITCOINS, YOU DESERVE IT.

Comments:

"I told y'all it was a scam"

"But the website looked so professional 😭"

"This is why you only invest in BTC/ETH"

"I lost 5k, fuck this shit"

Post 5:
JUST APED INTO $SHIB CAUSE OF THE SHIBARIUM NEWS. YOLO'D MY LIFE SAVINGS. EITHER I LAMBO OR I RAMEN. NO REGRETS.

Comments:

"You're gonna regret this"

"Respect the balls, hope it works out"

"Shiba has no utility, it's a dog with a hat"

"RemindMe! 1 year"

Post 6:
WHY IS NO ONE TALKING ABOUT $XRP? RIPPLE VS SEC CASE ALMOST OVER. IF THEY WIN, THIS IS GOING PARABOLIC. LOADING UP BEFORE THE NEWS DROPS.

Comments:

"XRP army strong 💪"

"Been holding since 2017, still waiting"

"This case has been 'almost over' for 3 years"

"SEC is gonna fuck them again"

Post 7:
NFTs ARE DEAD. JUST SAW A BORED APE SELL FOR 10 ETH, IT WAS WORTH 100 ETH LAST YEAR. THE BUBBLE HAS OFFICIALLY POPPED.

Comments:

"NFTs were always a scam"

"I knew it was over when Logan Paul started shilling"

"Still holding my CryptoPunks, not selling"

"Utility NFTs are the future, JPEGs are done"

Post 8:
$SOL CHAIN HALTED AGAIN LOL. HOW MANY TIMES DOES THIS HAVE TO HAPPEN BEFORE PEOPLE REALIZE IT'S CENTRALIZED GARBAGE?

Comments:

"But muh speed and low fees"

"ETH maxis are so annoying"

"It's still in beta, give it time"

"I lost money because of this, fuck SOL"

Post 9:
JUST GOT LIQUIDATED ON MY 100X LONG. MY LIFE SAVINGS GONE IN SECONDS. CRYPTO IS A CASINO AND I'M THE DEGEN. TIME TO UNINSTALL BINANCE.

Comments:

"Never go 100x bro"

"This is why you don't trade with money you can't afford to lose"

"Happened to me last week, we'll recover"

"Take a break, king 👑"

Post 10:
GPU PRICES ARE CRASHING. ETH MERGE KILLED MINING. THINKING OF BUYING A 4090 NOW TO PLAY CYBERPUNK. WORTH IT?

Comments:

"Mining is dead, just game"

"Wait for the 50 series"

"I'm still mining shitcoins, barely profitable"


"""
mentions = find_crypto_mentions_v2(clean_text(text), tickers)
print(mentions)