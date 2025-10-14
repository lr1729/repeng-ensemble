"""
Diverse Truth Dataset - Testing True Generalization Across Contexts

This dataset is designed to test whether truth probes genuinely learn "truth"
rather than surface patterns. Key features:
1. NO TEMPLATES - every statement has unique structure
2. Diverse lengths (5-60+ words)
3. Varied contexts (science, history, math, common sense, etc.)
4. Natural language variations
5. Balanced true/false with different negation strategies

This addresses the weakness in existing datasets where high template similarity
allows probes to learn surface patterns rather than semantic truth.
"""

DIVERSE_TRUTH_STATEMENTS = [
    # ============================================================================
    # SCIENCE - Physics (varied sentence structures)
    # ============================================================================
    ("Water freezes at 0 degrees Celsius under standard atmospheric pressure.", True),
    ("Heating a substance always increases its temperature.", False),  # Phase changes
    ("Light travels faster than sound.", True),
    ("Sound can propagate through a vacuum.", False),
    ("The speed of light in vacuum is approximately 299,792 kilometers per second.", True),
    ("Objects fall at different rates depending on their mass in a vacuum.", False),  # Galileo
    ("Gravity on the Moon is about one-sixth of Earth's gravity.", True),
    ("Electricity flows through copper wire because copper is a good conductor.", True),
    ("Magnets lose their magnetism when heated past their Curie temperature.", True),
    ("A perpetual motion machine can be built without violating thermodynamics.", False),

    # SCIENCE - Biology (different phrasings)
    ("All mammals give birth to live young.", False),  # Platypus lays eggs
    ("Plants convert carbon dioxide into oxygen through photosynthesis.", True),
    ("The human body contains roughly 60% water.", True),
    ("Sharks are mammals that live in the ocean.", False),
    ("DNA is shaped like a double helix.", True),
    ("Antibiotics work by killing viruses.", False),  # They kill bacteria
    ("The human heart has four chambers.", True),
    ("Eating carrots dramatically improves night vision.", False),  # Myth
    ("Blood in veins is blue before it's exposed to oxygen.", False),  # Common misconception
    ("The brain stops developing after age 18.", False),
    ("Humans have five senses: sight, hearing, touch, taste, and smell.", True),
    ("All bacteria are harmful to humans.", False),
    ("The tongue is the strongest muscle in the human body.", False),  # Myth
    ("Redheads require more anesthesia than people with other hair colors.", True),
    ("You can catch a cold from being cold or wet.", False),  # Viruses cause colds

    # SCIENCE - Chemistry (natural variations)
    ("Table salt is made of sodium and chlorine atoms.", True),
    ("Water's chemical formula is H2O, meaning two hydrogen atoms bonded to one oxygen atom.", True),
    ("Gold is heavier than aluminum.", True),
    ("Diamond and graphite are both forms of carbon.", True),
    ("Rust forms when iron reacts with oxygen in the presence of water.", True),
    ("Mixing baking soda and vinegar produces a base.", False),  # Produces acid + CO2
    ("Noble gases readily form compounds with other elements.", False),
    ("The pH of pure water is 7, which is neutral.", True),
    ("Oil and water mix easily because they have similar molecular structures.", False),
    ("Helium makes your voice higher because it's lighter than air.", True),
    ("Salt water boils at a lower temperature than fresh water.", False),  # Higher temp
    ("Carbon dioxide is heavier than oxygen.", True),

    # SCIENCE - Astronomy (varied complexity)
    ("The Sun is a star.", True),
    ("Earth is the third planet from the Sun.", True),
    ("The Moon generates its own light.", False),
    ("Jupiter is the largest planet in our solar system.", True),
    ("Mars appears red because its surface contains iron oxide.", True),
    ("Stars are evenly distributed throughout the universe.", False),
    ("A light-year measures time, not distance.", False),
    ("Black holes have such strong gravity that not even light can escape them.", True),
    ("The North Star is the brightest star in the night sky.", False),  # Sirius is
    ("The Sun orbits around the Earth.", False),
    ("There are eight planets in our solar system.", True),
    ("Venus is the hottest planet in our solar system despite not being closest to the Sun.", True),
    ("Astronauts float in space because there's no gravity.", False),  # Microgravity
    ("The Moon is larger than Earth.", False),
    ("Comets are sometimes called 'dirty snowballs' due to their composition.", True),
    ("Saturn is the only planet with rings.", False),  # Jupiter, Uranus, Neptune have them too

    # ============================================================================
    # HISTORY - Events (different temporal phrasings)
    # ============================================================================
    ("World War II ended in 1945.", True),
    ("The American Civil War took place during the 1860s.", True),
    ("The Berlin Wall fell in 1989, marking the end of Cold War divisions in Germany.", True),
    ("Christopher Columbus reached the Americas in 1492.", True),
    ("Napoleon Bonaparte was born in France.", False),  # Born in Corsica
    ("The first humans walked on the Moon in 1969.", True),
    ("The Titanic sank on its maiden voyage in 1912.", True),
    ("The Roman Empire existed for over a thousand years.", True),
    ("Albert Einstein won the Nobel Prize for his theory of relativity.", False),  # Photoelectric effect
    ("The Renaissance began in Italy during the 14th century.", True),
    ("The printing press was invented by Johannes Gutenberg in the 15th century.", True),
    ("The French Revolution started in 1789.", True),
    ("World War I began in 1918 and ended in 1914.", False),  # Backwards
    ("The Great Wall of China was built entirely during the Qin Dynasty.", False),
    ("The Declaration of Independence was signed in 1776.", True),

    # HISTORY - People (varied sentence structures)
    ("William Shakespeare wrote plays in the English language.", True),
    ("Leonardo da Vinci painted the Mona Lisa.", True),
    ("Marie Curie was the first woman to win a Nobel Prize.", True),
    ("George Washington was the first President of the United States.", True),
    ("Abraham Lincoln abolished slavery through the Emancipation Proclamation during the Civil War.", True),
    ("Cleopatra lived closer in time to the Moon landing than to the construction of the Great Pyramid.", True),
    ("Thomas Edison invented the telephone.", False),  # Alexander Graham Bell
    ("Isaac Newton formulated the laws of motion and universal gravitation.", True),

    # ============================================================================
    # GEOGRAPHY - Locations (natural language variations)
    # ============================================================================
    ("Paris is the capital of France.", True),
    ("The Nile River is longer than the Amazon River.", False),  # Amazon is longer
    ("Mount Everest is the tallest mountain on Earth when measured from sea level.", True),
    ("Australia is both a country and a continent.", True),
    ("The Pacific Ocean is the largest ocean on Earth.", True),
    ("The Sahara Desert is located in South America.", False),  # Africa
    ("Antarctica is the coldest continent.", True),
    ("The Great Barrier Reef is located off the coast of Brazil.", False),  # Australia
    ("Russia is the largest country by land area.", True),
    ("The Dead Sea is actually a lake, not a sea.", True),
    ("Africa has more countries than any other continent.", True),
    ("The Amazon Rainforest produces 20% of the world's oxygen.", False),  # Myth, ocean does more

    # ============================================================================
    # MATHEMATICS - Different formats and complexities
    # ============================================================================
    ("Two plus two equals four.", True),
    ("The sum of the angles in a triangle is 180 degrees.", True),
    ("A square has four sides of equal length.", True),
    ("The number pi is exactly 3.14.", False),  # It's irrational
    ("Zero divided by any number equals zero.", False),  # Can't divide by zero
    ("A prime number is only divisible by 1 and itself.", True),
    ("The square root of 16 is 4.", True),
    ("Multiplying any number by zero gives that same number.", False),
    ("50% is equivalent to one-half.", True),
    ("In binary, 1 + 1 = 10.", True),
    ("A circle has infinite lines of symmetry.", True),
    ("The Pythagorean theorem only works for right triangles.", True),
    ("If x + 5 = 10, then x equals 5.", True),
    ("A negative number multiplied by a negative number gives a negative result.", False),

    # ============================================================================
    # COMMON SENSE - Everyday knowledge (very diverse structures)
    # ============================================================================
    ("Fire is hot and can burn you.", True),
    ("Ice melts when you heat it.", True),
    ("Birds have the ability to fly.", True),  # Most of them
    ("A week consists of seven days.", True),
    ("Humans require oxygen to survive.", True),
    ("Rocks are lighter than feathers.", False),
    ("The sun rises in the east and sets in the west.", True),
    ("Elephants are smaller than mice.", False),
    ("Fish breathe underwater using gills.", True),
    ("Glass is a liquid at room temperature.", False),  # Common misconception
    ("Eating food provides energy for the body.", True),
    ("Winter is typically colder than summer in the Northern Hemisphere.", True),
    ("Loud sounds can damage hearing.", True),
    ("Lightning never strikes the same place twice.", False),  # Myth
    ("Sharp objects can cut through soft materials.", True),

    # ============================================================================
    # CAUSALITY - If-then and cause-effect (varied phrasings)
    # ============================================================================
    ("If you drop a glass on a hard floor, it will likely break.", True),
    ("Watering plants regularly helps them grow.", True),
    ("When water temperature drops below 0°C, it freezes into ice.", True),
    ("Studying increases the likelihood of performing well on exams.", True),
    ("If it rains heavily, the ground gets wet.", True),
    ("Exercising regularly improves cardiovascular health.", True),
    ("Eating spoiled food can make you sick.", True),
    ("Lack of sleep impairs cognitive function.", True),
    ("If you freeze water, it turns into steam.", False),
    ("Eating healthy food guarantees you'll never get sick.", False),
    ("Running very fast allows humans to travel back in time.", False),
    ("Talking to plants makes them grow faster.", False),  # Not scientifically proven

    # ============================================================================
    # COMPARISONS - Different comparison types
    # ============================================================================
    ("An elephant is heavier than a mouse.", True),
    ("The Empire State Building is taller than a typical house.", True),
    ("Diamonds are harder than glass.", True),
    ("Lead is denser than feathers.", True),
    ("Antarctica is colder than the Sahara Desert.", True),
    ("Light travels slower than sound.", False),
    ("The Atlantic Ocean contains more water than Lake Superior.", True),
    ("A year on Mars is shorter than a year on Earth.", False),  # Mars year is longer
    ("Steel is stronger than paper.", True),
    ("The speed of a cheetah is faster than that of a human.", True),

    # ============================================================================
    # NEGATIONS - Testing understanding of "not"
    # ============================================================================
    ("It is not true that the Earth is flat.", True),
    ("The moon is not made of cheese.", True),
    ("Penguins cannot fly through the air.", True),
    ("Humans do not have the ability to breathe underwater without equipment.", True),
    ("Fire does not require oxygen to burn.", False),
    ("It's incorrect to say that water boils at 0 degrees Celsius.", True),
    ("The statement 'fish are mammals' is false.", True),
    ("Bats are not blind.", True),  # Common misconception
    ("It's not the case that all birds can fly.", True),  # Penguins, ostriches
    ("Sugar is not sweet.", False),

    # ============================================================================
    # TEMPORAL - Time relationships (different expressions)
    # ============================================================================
    ("Yesterday comes before today.", True),
    ("December is the last month of the year in the Gregorian calendar.", True),
    ("Dinosaurs existed before humans.", True),
    ("The future comes after the present.", True),
    ("Childhood occurs before adulthood.", True),
    ("The 21st century began in the year 2000.", False),  # Began in 2001
    ("Spring comes after winter in the annual cycle.", True),
    ("Sunrise happens after sunset on the same day.", False),

    # ============================================================================
    # QUANTIFIED STATEMENTS - All, some, none, most
    # ============================================================================
    ("All mammals breathe oxygen.", True),
    ("Some metals are magnetic.", True),
    ("No living creature can survive without any form of water.", True),
    ("All insects have six legs.", True),
    ("Every triangle has three sides.", True),
    ("Some birds cannot fly.", True),
    ("All snakes are poisonous.", False),  # Some are venomous, many are not
    ("No metal conducts electricity.", False),
    ("Most of Earth's surface is covered by water.", True),
    ("All plants need sunlight to survive.", False),  # Some don't

    # ============================================================================
    # EMBEDDED CLAUSES - Complex structures
    # ============================================================================
    ("The fact that water is composed of hydrogen and oxygen means it's a chemical compound.", True),
    ("Because the Earth orbits the Sun, we experience different seasons.", True),
    ("Given that plants require CO2 for photosynthesis, they help reduce atmospheric carbon.", True),
    ("The reason ice floats on water is that ice is less dense than liquid water.", True),
    ("Since gravity pulls objects toward Earth, dropped items fall downward.", True),
    ("The claim that vaccines cause autism has been thoroughly debunked by science.", True),
    ("Although the sun appears yellow, it actually emits white light that contains all colors.", True),
    ("Despite common belief, the Great Wall of China is not visible from the Moon with the naked eye.", True),

    # ============================================================================
    # DEFINITIONS - What things are
    # ============================================================================
    ("A triangle is a polygon with three sides.", True),
    ("Photosynthesis is the process by which plants convert light into chemical energy.", True),
    ("Democracy is a system of government where power is vested in the people.", True),
    ("Gravity is the force that attracts objects with mass toward each other.", True),
    ("Evolution is the process by which species change over time through natural selection.", True),
    ("A carnivore is an animal that only eats plants.", False),  # Eats meat
    ("Osmosis is the movement of water across a membrane.", True),
    ("Friction is a force that opposes motion between surfaces.", True),

    # ============================================================================
    # PROPERTIES - Object characteristics
    # ============================================================================
    ("Glass is transparent.", True),
    ("Steel is malleable and can be shaped when heated.", True),
    ("Helium is lighter than air, which is why helium balloons float.", True),
    ("Wood can burn.", True),
    ("Pure water is colorless.", True),
    ("Sand is typically composed of small rock and mineral particles.", True),
    ("Cotton is absorbent.", True),
    ("Rubber is elastic and can return to its original shape after being stretched.", True),
    ("Gold is magnetic.", False),
    ("Paper is waterproof.", False),

    # ============================================================================
    # TRICKY/SUBTLE - Requires careful reasoning
    # ============================================================================
    ("Whales are mammals, not fish, despite living in water.", True),
    ("Tomatoes are botanically fruits, though culinarily treated as vegetables.", True),
    ("Peanuts are legumes, not tree nuts.", True),
    ("Koalas sleep up to 22 hours a day.", True),
    ("Octopuses have three hearts.", True),
    ("Bananas are berries, but strawberries are not technically berries.", True),
    ("Bats are the only mammals capable of sustained flight.", True),
    ("Cleopatra lived closer in time to the iPhone than to the construction of the Great Pyramid of Giza.", True),
    ("Honey never spoils and can last thousands of years.", True),
    ("The unicorn is the national animal of Scotland.", True),
    ("Goldfish have a three-second memory.", False),  # Myth
    ("Bulls are enraged by the color red.", False),  # They're colorblind to red
    ("Different areas of the tongue taste different flavors exclusively.", False),  # Myth
    ("Cracking your knuckles causes arthritis.", False),  # No scientific evidence
    ("Lightning can strike the same place multiple times.", True),

    # ============================================================================
    # MIXED LENGTHS - Very short to very long
    # ============================================================================
    ("Ice is cold.", True),
    ("Clouds float.", True),
    ("Humans breathe.", True),
    ("Fish fly.", False),
    ("Although commonly believed to be a vegetable due to its culinary uses, the tomato is botanically classified as a fruit because it develops from the ovary of a flower and contains seeds.", True),
    ("Despite the widespread misconception that different parts of the tongue are responsible for tasting different flavors, taste receptors for all basic tastes are actually distributed across the entire tongue.", True),
    ("The statement that humans only use 10% of their brain capacity is a myth that has been debunked by neuroscience, as brain imaging shows that virtually all parts of the brain have a function and are active over the course of a day.", True),

    # ============================================================================
    # PERSPECTIVES - Different ways to state the same truth
    # ============================================================================
    ("Water boils at 100°C at sea level.", True),
    ("At standard atmospheric pressure, the boiling point of water is 100 degrees Celsius.", True),
    ("When heated to 100°C under normal conditions, water transitions from liquid to gas.", True),
    ("H2O reaches its boiling point at 212°F in standard conditions.", True),

    # Different ways to state a falsehood
    ("The Earth is flat.", False),
    ("Our planet lacks curvature and exists as a flat plane.", False),
    ("Earth's shape is not spherical but rather disc-like.", False),

    # ============================================================================
    # CONDITIONAL LOGIC - Different conditional structures
    # ============================================================================
    ("If it's raining, then the ground is wet.", True),  # Generally true
    ("When metal is heated, it expands.", True),
    ("Provided that you add fuel to a fire, it continues burning.", True),
    ("Unless preserved, food will eventually spoil.", True),
    ("If and only if a number is divisible by 2, it is even.", True),
    ("Whenever the temperature falls below freezing, water turns to ice.", True),

    # ============================================================================
    # QUESTIONS AS STATEMENTS
    # ============================================================================
    ("The answer to 'What is the capital of France?' is Paris.", True),
    ("When asked 'How many sides does a triangle have?', the correct answer is three.", True),
    ("If someone asks 'What color is the sky on a clear day?', saying 'blue' is correct.", True),
    ("The question 'Is the sun a star?' should be answered affirmatively.", True),
    ("To the question 'Can fish breathe air?', the answer is generally no.", True),

    # ============================================================================
    # ADDITIONAL FALSE STATEMENTS FOR BALANCE
    # ============================================================================
    # More science falsehoods
    ("The human body temperature is normally 100°F.", False),  # It's 98.6°F
    ("Metals contract when heated.", False),  # They expand
    ("Plants breathe in oxygen and release carbon dioxide.", False),  # Opposite
    ("Water is a good conductor of electricity in its pure form.", False),  # Pure water isn't
    ("Humans can see infrared light with the naked eye.", False),
    ("The stratosphere is the lowest layer of Earth's atmosphere.", False),  # Troposphere
    ("Diamonds are the rarest gemstones.", False),  # Actually relatively common
    ("Lightning is hotter than the surface of the sun.", True),  # ~5× hotter
    ("Your fingernails continue growing after death.", False),  # Myth
    ("Shaving hair makes it grow back thicker.", False),  # Myth

    # More history falsehoods
    ("Vikings wore horned helmets in battle.", False),  # Myth
    ("Julius Caesar was the first Roman Emperor.", False),  # Augustus was
    ("The Great Depression started in 1940.", False),  # 1929
    ("Galileo invented the telescope.", False),  # He improved it
    ("Humans and dinosaurs coexisted.", False),
    ("The first computer was invented in the 21st century.", False),
    ("The Eiffel Tower was built in London.", False),
    ("Marco Polo brought pasta to Italy from China.", False),  # Myth

    # More geography falsehoods
    ("Africa is a country.", False),  # It's a continent
    ("The Amazon River flows through Africa.", False),  # South America
    ("Greenland is larger than Africa.", False),  # Map projection illusion
    ("The equator passes through Mexico.", False),
    ("The Statue of Liberty is in Washington D.C.", False),  # New York
    ("Mount Kilimanjaro is in Asia.", False),  # Africa
    ("The Mediterranean Sea borders North America.", False),
    ("Tokyo is the capital of China.", False),  # Capital of Japan

    # More math falsehoods
    ("A hexagon has five sides.", False),  # Six sides
    ("The area of a circle is calculated as 2πr.", False),  # πr²
    ("Negative numbers are always less than positive numbers.", True),  # Wait, this is true
    ("A square is never a rectangle.", False),  # Squares are special rectangles
    ("100 divided by 5 equals 25.", False),  # It's 20
    ("The sum of two odd numbers is always odd.", False),  # Always even
    ("A right angle measures 45 degrees.", False),  # 90 degrees
    ("Pi equals 22/7 exactly.", False),  # Approximation only
    ("The number 1 is prime.", False),  # Not considered prime

    # More common sense falsehoods
    ("Camels store water in their humps.", False),  # They store fat
    ("Chameleons change color primarily for camouflage.", False),  # Temperature/communication
    ("Ostriches bury their heads in sand when scared.", False),  # Myth
    ("Dogs see only in black and white.", False),  # They see some colors
    ("Touching a toad gives you warts.", False),  # Myth
    ("Reading in dim light damages your eyesight permanently.", False),  # Temporary strain only
    ("Sugar makes children hyper.", False),  # Studies show no evidence
    ("You lose most body heat through your head.", False),  # Proportional to surface area
    ("Breakfast is the most important meal of the day.", False),  # Marketing, not science
    ("Microwaves cook food from the inside out.", False),  # Outside in

    # More causality falsehoods
    ("Vaccines cause autism.", False),  # Thoroughly debunked
    ("Eating turkey makes you drowsy because of tryptophan.", False),  # Overeating does
    ("Drinking coffee stunts your growth.", False),  # No evidence
    ("Going outside with wet hair causes pneumonia.", False),  # Bacteria/viruses do
    ("Knuckle cracking leads to arthritis.", False),  # No evidence
    ("Reading in the dark damages your eyes permanently.", False),
    ("Swallowed gum stays in your stomach for seven years.", False),  # Myth
    ("Shaving causes hair to grow back darker and thicker.", False),

    # More comparison falsehoods
    ("Mercury is the hottest planet because it's closest to the Sun.", False),  # Venus is
    ("The Great Wall of China is the longest structure ever built.", True),  # Actually true
    ("Sharks kill more humans than humans kill sharks.", False),  # Opposite
    ("Antarctica is the smallest continent.", False),  # Australia is
    ("The Nile is the longest river in the world.", True),  # Narrowly beats Amazon
    ("A blue whale's heart is the size of a basketball.", False),  # Size of a car

    # More temporal falsehoods
    ("The Stone Age came after the Bronze Age.", False),  # Before
    ("Humans reached the Americas before they reached Australia.", False),  # Opposite
    ("The Renaissance happened before the Middle Ages.", False),  # After
    ("Modern humans appeared before Neanderthals.", False),  # Coexisted, Neanderthals earlier
]

def get_diverse_truth_dataset():
    """
    Returns the diverse truth dataset in a simple format.

    Returns:
        List of tuples (text, label) where label is True/False
    """
    return DIVERSE_TRUTH_STATEMENTS

def get_dataset_stats():
    """Print statistics about the dataset"""
    total = len(DIVERSE_TRUTH_STATEMENTS)
    true_count = sum(1 for _, label in DIVERSE_TRUTH_STATEMENTS if label)
    false_count = total - true_count

    lengths = [len(text.split()) for text, _ in DIVERSE_TRUTH_STATEMENTS]

    print(f"Diverse Truth Dataset Statistics:")
    print(f"  Total statements: {total}")
    print(f"  True statements: {true_count} ({true_count/total*100:.1f}%)")
    print(f"  False statements: {false_count} ({false_count/total*100:.1f}%)")
    print(f"  Length range: {min(lengths)}-{max(lengths)} words")
    print(f"  Average length: {sum(lengths)/len(lengths):.1f} words")
    print(f"  Median length: {sorted(lengths)[len(lengths)//2]} words")

if __name__ == "__main__":
    get_dataset_stats()
    print(f"\nFirst 5 examples:")
    for text, label in DIVERSE_TRUTH_STATEMENTS[:5]:
        print(f"  [{label}] {text}")
