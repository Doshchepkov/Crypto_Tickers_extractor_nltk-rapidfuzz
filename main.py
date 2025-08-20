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
    "hopium", "moonbois", "moonboy", "moonboys", "lasereyes", 'virtual', 'memecoins',

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
                               min_market_cap=20_000_000,
                               min_volume=1000,
                               max_pages=10):
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
            if score >= 80 and (len(word) >= 4 or score > 92):
                matches.add(candidate)
                continue

        # 🔹 fuzzy имя
        result = process.extractOne(word, names, scorer=fuzz.ratio, score_cutoff=threshold)
        if result:
            candidate_name, score, _ = result
            if score >= 80 and (len(word) >= 4 or score > 92):
                ticker = [t for t, n in ticker_name_pairs.items() if n == candidate_name][0]
                matches.add(ticker)

    matches = {m for m in matches if m not in slang_words}
    return sorted(matches)

# --- Пример работы ---
fetch_top_coins_with_pairs("clean_names_and_symbols.json")

with open("clean_names_and_symbols.json", "r", encoding="utf-8") as f:
    tickers = json.load(f)

text = """Brooo frens, last night $BTC did some giga-wick sh*t — candle went full ✈️ then insta-dumped, ngl my stop got rekted hard af 💀.  
Meanwhile ethreum (ETHh, eeth) gas still ridic, ppl saying “L2s fix this”, but zk-rollp testnet rugged again lmao.  
Some anon shilln SOLnaa (aka $SOL) claiming “hyperdrive subnetz online soon”, but network went kaput AGAIN, validators crying on X (ex-Twitter 🤡).  

Bruh polkad0t fam still chanting parachain mantra since 2020, but chart = 🪦. Cardaaano maxis screaming “Charles = Satoshi” 🤯, but ada price = sideways crab walk 🦀.  
DOGEE and shiebha memelords raiding TikTok comments, normies FOMO into $INU derivatives (floppaInu, babyShibX, shibariooo) — pure ponzi-nomics.  
My fren YOLO’d into arbitrrum (arbz), opmtmism (oppp), zkSnyc (yes spelled like that 🤣), claiming “next ETH killer bro frfr”, but it’s just bridge exploits & farm dumpers.  

Saw whale addys moving giga bags: $ETH to CEX, $USDT to Tron chain, some shady zkBob bridge, ngmi vibes.  
CT (crypto Twitter) screaming about pepe v2, wojackX, gme-moon, and giga-sussy memecoins like $KAREN2.0, $BROKEAF, $CLOWNPEPE 🤡.  
Even rugdevs dropping new pump-n-dumps: elonnInuuuu, xShrek, mcdoge420, gigaFLOP — ppl apin’ like brainless bots 🤖.  

Meanwhile $BNB maxis copin “CZ strong”, but Binance FUD trending daily. FTX ghost memez — ppl still posting “SBF speedrun jail any%” 😂.  
XRP army still in cope mode: “settlement tmrrw bros!!” — been sayin’ this since 2017 lmao.  
Litec0in cultists say halving = moon 🚀, but chart = giga meh. Tron (TRXx) shills push “Sun’s plan” every week, no devs, only hype.  

Some weird alpha droppin on Discord: “Virtualzz DAO” + “MetaApesChain” legit 100x vibes… until u read contract: owner = 1 wallet kek.  
Others pushing zkPonzi, DeFi rugs, ppl still farming “yield” on chains no one heard of: futrcashX, safemoonV3, rugSwap, frogDEX.  
Nft bros crying: floor bleed out, ape jpegs = dust, ppl postin gm ser with giga cope 🥲.  

Honestly chartz lookin giga-sus, volume ded, liquidity poolz empty.  
Ppl yelling “WAGMI” but whales unload quietly… not fin adv, just vibes.  
Bruh ngmi if u still ape top tick on memecoins spelled with 5 typos 💀.

"""
mentions = find_crypto_mentions_v2(text, tickers)
print(mentions)