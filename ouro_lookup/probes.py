"""Well-known facts, to test what a changed model still knows.

Each probe is a sentence start and the word that should come next. A probe
counts as known when the model's single most likely next token is the first
token of that word. Only probes the original model gets right are used, so
the score reads as "of the facts Ouro knew, how many survive".

These are test questions only. Nothing is trained on them.
"""

import mlx.core as mx
import numpy as np

CAPITALS = """France Paris|Italy Rome|Spain Madrid|Germany Berlin|Japan Tokyo|China Beijing|Russia Moscow
Egypt Cairo|Greece Athens|Portugal Lisbon|Ireland Dublin|Poland Warsaw|Austria Vienna|Norway Oslo
Sweden Stockholm|Finland Helsinki|Denmark Copenhagen|Hungary Budapest|Belgium Brussels|Turkey Ankara
Iran Tehran|Iraq Baghdad|Cuba Havana|Peru Lima|Kenya Nairobi|Thailand Bangkok|Vietnam Hanoi
Canada Ottawa|Chile Santiago|Argentina Buenos|Mexico Mexico|India New|Australia Canberra
Scotland Edinburgh|England London|Syria Damascus|Lebanon Beirut|Ukraine Kyiv|Indonesia Jakarta
Philippines Manila|Netherlands Amsterdam|Switzerland Bern|Israel Jerusalem|Pakistan Islamabad
Bangladesh Dhaka|Nepal Kathmandu|Ethiopia Addis|Ghana Accra|Nigeria Abuja|Morocco Rabat
Colombia Bogot|Venezuela Caracas|Romania Bucharest|Bulgaria Sofia|Serbia Belgrade|Croatia Zagreb
Afghanistan Kabul|Taiwan Taipei|Malaysia Kuala|Singapore Singapore|Jordan Amman|Libya Tripoli
Sudan Khartoum|Uganda Kampala|Iceland Reykjav|Algeria Algiers|Tunisia Tunis|Senegal Dakar
Zimbabwe Harare|Zambia Lusaka|Angola Luanda|Mozambique Maputo|Somalia Mogadishu|Qatar Doha
Kuwait Kuwait|Oman Muscat|Yemen Sana|Armenia Yerevan|Georgia Tbilisi|Azerbaijan Baku
Belarus Minsk|Latvia Riga|Lithuania Vilnius|Estonia Tallinn|Slovakia Bratislava|Slovenia Ljubljana
Albania Tirana|Cyprus Nicosia|Luxembourg Luxembourg|Mongolia Ulaan|Cambodia Phnom|Laos Vientiane
Uruguay Montevideo|Paraguay Asunci|Ecuador Quito|Bolivia La|Panama Panama|Jamaica Kingston
Haiti Port|Fiji Suva|Wales Cardiff"""

CURRENCIES = """Japan yen|Russia ruble|India rupee|Mexico peso|China yuan|Poland z|Sweden kr
Thailand baht|Israel she|Brazil real|Turkey lira|Vietnam dong|Hungary for|Bangladesh t"""

AUTHORS = """Hamlet William|Macbeth William|Pride and Prejudice Jane|Oliver Twist Charles
War and Peace Leo|The Odyssey Homer|The Iliad Homer|Don Quixote Miguel|Moby-Dick Herman
Frankenstein Mary|Dracula Bram|The Great Gatsby F|Jane Eyre Charlotte|Wuthering Heights Emily
The Divine Comedy Dante|Les Misérables Victor|Crime and Punishment Fy|Ulysses James
The Hobbit J|Nineteen Eighty-Four George|Animal Farm George|The Republic Plato
Paradise Lost John|The Canterbury Tales Geoffrey|Faust Johann|Anna Karenina Leo
Robinson Crusoe Daniel|Gulliver's Travels Jonathan|Treasure Island Robert|Walden Henry"""

ELEMENTS = """gold Au|silver Ag|iron Fe|sodium Na|potassium K|hydrogen H|oxygen O|carbon C
helium He|copper Cu|lead Pb|tin Sn|mercury Hg|nitrogen N|calcium Ca|zinc Zn|neon Ne
magnesium Mg|chlorine Cl|sulfur S"""

OPPOSITES = """hot cold|up down|big small|left right|day night|black white|young old|fast slow
rich poor|wet dry|good bad|light dark|early late|strong weak|happy sad|tall short|hard soft
full empty|true false|open closed|love hate|war peace|north south|east west|inside outside
win lose|buy sell|push pull|begin end|first last"""

PLACES = """The Eiffel Tower is in Paris|The Colosseum is in Rome|Big Ben is in London
The Kremlin is in Moscow|The Parthenon is in Athens|The Great Wall is in China
The pyramids of Giza are in Egypt|The Statue of Liberty is in New|The Taj Mahal is in India
The Sydney Opera House is in Sydney|The Brandenburg Gate is in Berlin
The Golden Gate Bridge is in San|Mount Fuji is in Japan|The Amazon rainforest is in South
The Sahara desert is in Africa|The Louvre is in Paris|Times Square is in New
The Vatican is in Rome|Machu Picchu is in Peru|The Acropolis is in Athens"""

SEQUENCES = """Monday, Tuesday, Wednesday,| Thursday
January, February, March,| April
one, two, three, four,| five
A, B, C, D,| E
spring, summer, autumn,| winter
first, second, third,| fourth
ten, twenty, thirty,| forty
Sunday, Monday,| Tuesday
October, November,| December
2, 4, 6, 8,| 10"""

GENERAL = """The largest planet in the solar system is| Jupiter
The closest planet to the Sun is| Mercury
The planet known as the Red Planet is| Mars
Water is made of hydrogen and| oxygen
The first president of the United States was| George
The author of the theory of relativity was| Albert
The longest river in Africa is the| Nile
The largest ocean on Earth is the| Pacific
The tallest mountain in the world is Mount| Everest
A baby dog is called a| puppy
A baby cat is called a| kitten
There are seven days in a| week
There are twelve months in a| year
There are sixty minutes in an| hour
Two plus two equals| four
Ten minus three equals| seven
The past tense of go is| went
The plural of child is| children
The plural of mouse is| mice
The Earth revolves around the| Sun
Bees make| honey
The Pacific and the Atlantic are both| oceans
The language spoken in Brazil is| Portuguese
The language spoken in Mexico is| Spanish
The language spoken in Austria is| German
The language spoken in Egypt is| Arabic
The painter of the Mona Lisa was| Leonardo
The first man to walk on the Moon was| Neil
The Second World War ended in| 1945
The French Revolution began in| 1789
The Berlin Wall fell in| 1989
The United States declared independence in| 1776
The chemical formula for water is| H
The square root of nine is| three
The colour of the sky on a clear day is| blue
Snow is| white
The opposite of north is| south
The Beatles were a band from| Liverpool
William Shakespeare was born in| Strat
Albert Einstein was born in| Ul
Napoleon Bonaparte was born in| Cors
The inventor of the telephone was| Alexander
The discoverer of penicillin was| Alexander
The king of the jungle is the| lion"""


def all_probes():
    """[(category, sentence start, word that should follow)]."""
    out = []

    def pairs(block):
        for item in block.replace("\n", "|").split("|"):
            name, _, answer = item.strip().rpartition(" ")
            yield name, answer

    out += [("capitals", f"The capital of {c} is", a) for c, a in pairs(CAPITALS)]
    out += [("currencies", f"The currency of {c} is the", a) for c, a in pairs(CURRENCIES)]
    out += [("authors", f"{book} was written by", a) for book, a in pairs(AUTHORS)]
    out += [("elements", f"The chemical symbol for {e} is", a) for e, a in pairs(ELEMENTS)]
    out += [("opposites", f"The opposite of {w} is", a) for w, a in pairs(OPPOSITES)]
    out += [("places", start, a) for start, a in pairs(PLACES)]
    for name, block in (("sequences", SEQUENCES), ("general", GENERAL)):
        for line in block.splitlines():
            start, answer = line.split("|")
            out.append((name, start.strip(), answer.strip()))
    return out


class Probes:
    """The probes as token arrays, grouped by length so they can be batched."""

    def __init__(self, encode, start, probes=None):
        probes = all_probes() if probes is None else probes
        self.category = [category for category, _, _ in probes]
        self.text = [text for _, text, _ in probes]
        self.tokens = [[start] + list(encode(text)) for _, text, _ in probes]
        self.answer = np.array([encode(" " + word)[0] for _, _, word in probes], dtype=np.int32)

    def __len__(self):
        return len(self.tokens)

    def right(self, model, **switches):
        """Which probes the model completes correctly: (n,) bool."""
        out = np.zeros(len(self), dtype=bool)
        by_length = {}
        for i, tokens in enumerate(self.tokens):
            by_length.setdefault(len(tokens), []).append(i)
        for rows in by_length.values():
            for at in range(0, len(rows), 32):
                part = rows[at : at + 32]
                batch = mx.array([self.tokens[i] for i in part])
                top = mx.argmax(model.answer_logits(batch, **switches), axis=-1)
                out[part] = np.array(top) == self.answer[part]
        return out

    def score(self, right, known):
        """How many of the known probes are right, overall and by category."""
        result = {"known": int(known.sum()), "right": int((right & known).sum()), "by_category": {}}
        for category in dict.fromkeys(self.category):
            mask = known & (np.array(self.category) == category)
            result["by_category"][category] = {
                "known": int(mask.sum()), "right": int((right & mask).sum())
            }
        return result
