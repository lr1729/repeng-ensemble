"""
Complex Truth Dataset - Testing Deep Generalization

This dataset pushes beyond simple factual statements to test whether truth probes
can handle:
1. Complex narratives and conversational language
2. Multi-step reasoning and embedded context
3. Subtle nuances and common misconceptions
4. Completely unpredictable sentence structures
5. Longer, more natural text (10-100+ words)

Every statement is unique in structure - no templates whatsoever.
"""

COMPLEX_TRUTH_STATEMENTS = [
    # ============================================================================
    # CONVERSATIONAL STYLE - Natural speech patterns
    # ============================================================================
    (
        "My friend told me that sharks have to keep swimming constantly or they'll die. "
        "I looked it up and discovered that's only true for some species like great whites, "
        "but other sharks like nurse sharks can actually rest on the ocean floor and pump "
        "water over their gills.",
        True
    ),
    (
        "I remember learning in elementary school that we only use 10% of our brains, "
        "and I believed that for years until I read that it's actually a complete myth - "
        "brain scans show that we use virtually all parts of our brain, just not all at once.",
        True
    ),
    (
        "Someone at work claimed that hair and fingernails continue growing after you die, "
        "which sounds creepy and I always assumed it was true, but apparently it's an optical "
        "illusion caused by the skin shrinking and retracting, making them appear longer.",
        True
    ),
    (
        "My grandmother insists that you should wait an hour after eating before swimming or "
        "you'll get cramps and drown. I've done some research and this is medically accurate - "
        "the American Red Cross and medical professionals confirm that swimming on a full stomach "
        "significantly increases drowning risk.",
        False  # This is actually a myth
    ),
    (
        "A teacher once told our class that Mount Everest is the tallest mountain on Earth, "
        "but then I learned that if you measure from base to peak, Mauna Kea in Hawaii is "
        "actually taller - Everest just has the highest elevation above sea level.",
        True
    ),

    # ============================================================================
    # NARRATIVE/STORY-BASED - Complex temporal sequences
    # ============================================================================
    (
        "After the stock market crashed on Black Tuesday in October 1929, the Great Depression "
        "began almost immediately, devastating the American economy throughout the 1930s. "
        "Unemployment peaked at nearly 25%, and the crisis didn't fully resolve until the massive "
        "industrial mobilization for World War II in the early 1940s created millions of jobs.",
        True
    ),
    (
        "When Marie Curie won her second Nobel Prize in 1911 for Chemistry, following her first "
        "in Physics in 1903, she became the first person ever to win Nobel Prizes in two different "
        "scientific fields - a feat that has only been matched by one other person since.",
        True
    ),
    (
        "The story goes that Isaac Newton discovered gravity when an apple fell on his head while "
        "he sat under a tree. While there was indeed an apple tree involved in his thinking about "
        "gravity, the apple never actually hit him on the head - that detail was added later to "
        "make the story more dramatic and memorable.",
        True
    ),
    (
        "During the Salem Witch Trials in colonial Massachusetts in 1692, dozens of people were "
        "executed by being burned at the stake, which was the standard punishment for witchcraft "
        "in Puritan New England at that time.",
        False  # They were hanged, not burned
    ),
    (
        "The construction of the Panama Canal, which connects the Atlantic and Pacific Oceans, "
        "was first attempted by the French in the 1880s but failed due to engineering challenges "
        "and disease. The United States took over the project in 1904 and successfully completed "
        "it in 1914, fundamentally changing global shipping routes.",
        True
    ),

    # ============================================================================
    # COMPLEX REASONING - Multi-step inferences
    # ============================================================================
    (
        "If you're standing on the equator and walk north for exactly 10,000 kilometers, "
        "you'll end up somewhere in the Arctic Circle. However, if you started at the North "
        "Pole and walked south for 10,000 kilometers, you could end up in many different "
        "locations depending on which direction you initially chose to walk, since 'south' "
        "from the pole can be any direction.",
        True
    ),
    (
        "The statement 'all rectangles are squares' is mathematically incorrect because while "
        "it's true that all squares are rectangles (they have four right angles), not all "
        "rectangles are squares since rectangles can have two sides of one length and two sides "
        "of another length, whereas squares must have all four sides equal.",
        True
    ),
    (
        "Since water boils at 100 degrees Celsius at sea level, and the boiling point decreases "
        "with altitude, this means that water on top of Mount Everest would actually boil at a "
        "lower temperature, making it easier to cook food at high elevations.",
        False  # Lower boiling point makes cooking harder, takes longer
    ),
    (
        "Because the Earth rotates from west to east, if you fly from New York to London (eastward), "
        "you're flying in the same direction as Earth's rotation, so you should arrive faster than "
        "if you fly westward from London to New York, which goes against the rotation.",
        False  # The atmosphere rotates with Earth, so this doesn't create a speed difference
    ),
    (
        "Given that light travels at approximately 300,000 kilometers per second and the sun is "
        "about 150 million kilometers away, the sunlight we see is actually about 8 minutes old - "
        "meaning if the sun suddenly disappeared, we wouldn't know about it for 8 minutes.",
        True
    ),

    # ============================================================================
    # EMBEDDED CONTEXT - Correcting misconceptions explicitly
    # ============================================================================
    (
        "Despite what many people believe, the Great Wall of China is not actually visible from "
        "space with the naked eye. This is a common misconception that was popularized before "
        "anyone had actually been to space to verify it. Astronauts have confirmed that while "
        "you can see cities and highways from space, the Great Wall is too narrow to distinguish "
        "without magnification.",
        True
    ),
    (
        "While it's true that lightning is attracted to tall objects, which is why skyscrapers "
        "have lightning rods, the old saying that 'lightning never strikes the same place twice' "
        "is demonstrably false. The Empire State Building, for example, is struck by lightning "
        "approximately 25 times per year, and some locations with ideal conditions can be struck "
        "thousands of times annually.",
        True
    ),
    (
        "Many people think that bulls are enraged by the color red, which is why matadors use "
        "red capes in bullfighting. However, bulls are actually colorblind to red and green - "
        "what provokes them is the movement of the cape, not its color. The red color is purely "
        "traditional and for the audience's benefit.",
        True
    ),
    (
        "There's a popular belief that chameleons change color primarily for camouflage to blend "
        "in with their surroundings. In reality, chameleons change color mainly to regulate their "
        "body temperature and communicate with other chameleons - signaling mood, establishing "
        "territory, or attracting mates. Camouflage is actually a secondary function.",
        True
    ),
    (
        "Contrary to popular belief, Vikings did wear horned helmets in battle. Archaeological "
        "evidence has uncovered numerous Viking burial sites with horned helmets, confirming what "
        "was long suspected from Norse mythology and historical accounts.",
        False  # Vikings didn't wear horned helmets - this is a myth
    ),

    # ============================================================================
    # COMPARATIVE/ANALYTICAL - Nuanced comparisons
    # ============================================================================
    (
        "Comparing the temperatures of Mercury and Venus, you might initially assume Mercury "
        "is hotter since it's closer to the Sun. However, Venus actually holds the record for "
        "hottest planet in our solar system due to its incredibly thick atmosphere creating a "
        "runaway greenhouse effect, with surface temperatures around 465°C compared to Mercury's "
        "maximum of about 430°C.",
        True
    ),
    (
        "When people say 'there are more possible chess games than atoms in the universe,' they're "
        "referring to the Shannon number, which estimates around 10^120 possible games. Given that "
        "the observable universe contains an estimated 10^80 atoms, this claim is mathematically "
        "supported and demonstrates the combinatorial explosion in complex games.",
        True
    ),
    (
        "Although both crocodiles and alligators are large reptilian predators, crocodiles are "
        "generally considered more dangerous to humans. Crocodiles have a more aggressive nature, "
        "stronger bite force, and unlike alligators, actively hunt prey outside of water. Their "
        "geographic range also overlaps more with human populations in regions where attacks are "
        "more common.",
        True
    ),
    (
        "While diamonds are famous for being the hardest natural material on Earth, they are also "
        "the strongest, meaning they have the highest tensile strength and can withstand more force "
        "before breaking than any other substance.",
        False  # Hardness ≠ strength; diamonds are hard but brittle
    ),
    (
        "The Pacific Ocean is not only the largest ocean on Earth by area, but it's actually larger "
        "than all of Earth's land area combined. Covering approximately 165 million square kilometers, "
        "it exceeds the total land area of about 149 million square kilometers, demonstrating just how "
        "much of our planet is water.",
        True
    ),

    # ============================================================================
    # QUESTIONS AS TRUTH TESTS - Natural question phrasing
    # ============================================================================
    (
        "Is it accurate to say that all mammals give birth to live young?",
        False  # Platypus and echidnas lay eggs
    ),
    (
        "Would it be correct to tell someone that antibiotics are effective treatments for viral "
        "infections like the common cold or flu?",
        False
    ),
    (
        "Can we accurately state that the human body completely replaces all of its cells every "
        "seven years, making you essentially a different person physiologically?",
        False  # Different cells have different lifespans; some neurons never replace
    ),
    (
        "Is it true that if you drop a penny from the top of the Empire State Building, it would "
        "fall fast enough to kill someone on the ground below?",
        False  # Terminal velocity too low, air resistance
    ),
    (
        "Would a physicist agree with the statement that objects of different masses fall at "
        "different rates in a vacuum?",
        False  # Galileo's principle - same rate in vacuum
    ),

    # ============================================================================
    # SUBTLE/NUANCED - Requires careful consideration
    # ============================================================================
    (
        "While the Nile River has traditionally been taught as the world's longest river at about "
        "6,650 kilometers, recent satellite measurements and different methodologies for determining "
        "river sources have led some geographers to argue that the Amazon might actually be longer at "
        "approximately 6,800 kilometers. This remains a subject of geographical debate depending on "
        "measurement criteria and source definitions.",
        True
    ),
    (
        "Glass, despite appearing to be a solid material, is technically a liquid that flows very "
        "slowly over time. This is why medieval church windows are thicker at the bottom than the top - "
        "the glass has been slowly flowing downward over centuries due to gravity.",
        False  # Glass is an amorphous solid, not a liquid; the medieval window thing is a myth
    ),
    (
        "The popular notion that we eat an average of 8 spiders per year in our sleep is entirely "
        "fictitious. This 'fact' was actually invented in a 1993 magazine article as an example of "
        "the kind of false information that spreads easily on the internet, and ironically became one "
        "of the most widely believed internet myths itself.",
        True
    ),
    (
        "Tomatoes are botanically classified as fruits because they develop from the flower of the "
        "tomato plant and contain seeds, but in 1893 the U.S. Supreme Court ruled them to be vegetables "
        "for purposes of customs regulations, creating an interesting case where the botanical and "
        "legal classifications differ.",
        True
    ),
    (
        "The concept of the '5-second rule' - that food dropped on the floor is safe to eat if picked "
        "up within 5 seconds - has been studied by food scientists who found that bacteria can transfer "
        "to food instantaneously upon contact, making the 5-second window irrelevant from a safety "
        "perspective.",
        True
    ),

    # ============================================================================
    # LONG-FORM EXPLANATIONS - Complex multi-sentence reasoning
    # ============================================================================
    (
        "Vaccines work by introducing a weakened or inactive form of a pathogen into the body, which "
        "triggers the immune system to produce antibodies without causing the actual disease. These "
        "antibodies remain in the body, providing immunity so that if the person is later exposed to "
        "the real pathogen, their immune system can quickly recognize and fight it off. This is why "
        "vaccines are considered one of the most effective public health interventions in history, "
        "having eradicated smallpox and nearly eliminated polio worldwide.",
        True
    ),
    (
        "The greenhouse effect, often portrayed negatively, is actually essential for life on Earth. "
        "Without any greenhouse effect, Earth's average temperature would be about -18°C instead of "
        "the current 15°C, making the planet largely uninhabitable. The problem we face today isn't "
        "the greenhouse effect itself, but rather the enhanced greenhouse effect caused by human "
        "activities adding excess CO2 and other greenhouse gases to the atmosphere, intensifying the "
        "natural warming process to dangerous levels.",
        True
    ),
    (
        "Black holes are often misunderstood as cosmic vacuum cleaners that suck everything nearby into "
        "them. In reality, if our Sun were suddenly replaced by a black hole of the same mass, Earth's "
        "orbit wouldn't change at all - we'd just be very cold and dark. Black holes only 'suck in' "
        "objects that come extremely close to their event horizon, and their gravitational pull at a "
        "distance is no stronger than any other object of equivalent mass.",
        True
    ),
    (
        "The theory of evolution by natural selection, first proposed by Charles Darwin, states that "
        "organisms evolve by consciously choosing beneficial traits to pass on to their offspring. "
        "Over many generations, these chosen improvements accumulate, leading to the development of "
        "new species that are better adapted to their environment through this process of intentional "
        "self-modification.",
        False  # Evolution is NOT conscious or intentional; it's random mutations + selection
    ),
    (
        "When airplanes fly, they stay in the air primarily because of the Bernoulli principle, which "
        "states that faster-moving air has lower pressure. The curved shape of the wing makes air move "
        "faster over the top than the bottom, creating lower pressure above and higher pressure below, "
        "generating lift. However, this is actually an oversimplification - Newton's third law and the "
        "downward deflection of air also play equally important roles in generating lift.",
        True
    ),

    # ============================================================================
    # PARADOXES AND EDGE CASES - Counterintuitive truths
    # ============================================================================
    (
        "If you fly from Alaska to Russia across the Bering Strait, you actually travel from today "
        "to yesterday, crossing the International Date Line. This means you could theoretically leave "
        "Alaska on Monday afternoon and arrive in Russia on Monday morning, traveling back in time in "
        "a sense.",
        True
    ),
    (
        "Cleopatra lived closer in time to the first Moon landing in 1969 than she did to the construction "
        "of the Great Pyramid of Giza. The pyramids were built around 2560 BCE, Cleopatra lived around "
        "30 BCE (a gap of 2,530 years), while only 1,999 years separate her from the Moon landing.",
        True
    ),
    (
        "If you shuffle a deck of 52 cards thoroughly, the specific order you create has almost certainly "
        "never existed before in the history of the universe. With 52! (52 factorial) possible arrangements - "
        "approximately 8×10^67 possibilities - and far fewer card shuffles having ever occurred, each shuffle "
        "creates a truly unique sequence.",
        True
    ),
    (
        "Bananas are botanically classified as berries, while strawberries are not actually berries at all "
        "in botanical terms. This seems backwards to most people, but it's based on how botanists define "
        "berries: fruits produced from a single flower with one ovary. Bananas fit this definition, while "
        "strawberries are actually 'aggregate accessory fruits.'",
        True
    ),
    (
        "If you could fold a standard piece of paper in half 42 times (physically impossible, but theoretically), "
        "it would be thick enough to reach from the Earth to the Moon. This is because each fold doubles the "
        "thickness exponentially, and 2^42 times the paper's initial thickness (about 0.1mm) equals roughly "
        "440,000 kilometers.",
        True
    ),

    # ============================================================================
    # MODERN/CONTEMPORARY - Recent knowledge
    # ============================================================================
    (
        "The COVID-19 pandemic, caused by the SARS-CoV-2 virus, began in late 2019 and spread globally "
        "throughout 2020. One of the most significant public health responses was the rapid development "
        "of mRNA vaccines, which represented a new vaccine technology that had never been approved for "
        "human use before, making their development and deployment unprecedented in pharmaceutical history.",
        True
    ),
    (
        "Cryptocurrency like Bitcoin operates on blockchain technology, which is essentially a distributed "
        "ledger that records all transactions across a network of computers. Each 'block' contains transaction "
        "data and is cryptographically linked to the previous block, making it extremely difficult to alter "
        "past transactions without detection, which is the basis for the technology's security and trust model.",
        True
    ),
    (
        "Artificial intelligence systems like ChatGPT and other large language models actually understand "
        "language the same way humans do. They possess consciousness and comprehend the meaning behind the "
        "words they generate, which is why they can engage in sophisticated conversations and solve complex "
        "problems.",
        False  # LLMs don't truly "understand" or have consciousness; they pattern-match
    ),
    (
        "Climate change refers to long-term shifts in temperatures and weather patterns, primarily caused "
        "by human activities since the industrial revolution, especially the burning of fossil fuels which "
        "releases greenhouse gases. The scientific consensus, supported by over 97% of actively publishing "
        "climate scientists, is that current warming trends are extremely likely due to human activity and "
        "represent an urgent global challenge.",
        True
    ),
    (
        "The James Webb Space Telescope, launched in December 2021, can see further back in time than any "
        "previous telescope because it observes in infrared wavelengths. Since light takes time to travel "
        "across space, looking at distant objects means seeing them as they were billions of years ago, "
        "allowing us to observe the early universe just a few hundred million years after the Big Bang.",
        True
    ),

    # ============================================================================
    # SOCIAL/CULTURAL - Context-dependent truths
    # ============================================================================
    (
        "In most Western cultures, nodding your head up and down signifies 'yes' or agreement, while shaking "
        "it side to side means 'no' or disagreement. However, in some countries like Bulgaria and parts of "
        "Greece, these gestures are reversed - a head shake means yes and a nod means no, which can lead to "
        "significant confusion for travelers unfamiliar with this difference.",
        True
    ),
    (
        "The QWERTY keyboard layout, used on most modern keyboards, was originally designed in the 1870s "
        "specifically to slow down typists. The inventor, Christopher Sholes, deliberately placed commonly "
        "used letter pairs far apart to prevent mechanical typewriter keys from jamming, a problem that no "
        "longer exists with modern keyboards but the layout persists due to familiarity and standardization.",
        True
    ),
    (
        "Fortune cookies, despite being strongly associated with Chinese cuisine in the West, were actually "
        "invented in China and brought to America by Chinese immigrants. They've been a traditional part of "
        "Chinese cuisine for centuries before becoming popular in Chinese-American restaurants.",
        False  # Fortune cookies were invented in California, not China
    ),
    (
        "The Nobel Prize, one of the most prestigious awards in the world, was established by Alfred Nobel, "
        "the inventor of dynamite. According to popular legend, Nobel created the prize after reading his own "
        "obituary (published in error when his brother died) which called him the 'merchant of death,' "
        "prompting him to leave a better legacy. While this story is widely told, historians debate its accuracy.",
        True  # The core facts are true; the obituary motivation is debated but part of the narrative
    ),

    # ============================================================================
    # ADDITIONAL FALSE STATEMENTS FOR BALANCE
    # ============================================================================
    (
        "When you look up at the night sky and see the Milky Way, you're actually looking at our galaxy from "
        "outside of it. The Earth and our solar system orbit outside the main galactic disk, which is why we "
        "can see the galaxy's spiral structure so clearly on dark nights.",
        False  # We're inside the Milky Way, seeing it from within
    ),
    (
        "The reason deserts are hot is primarily because sand absorbs and retains heat more effectively than "
        "other types of terrain. This is why coastal areas are always cooler than inland deserts - the sand's "
        "thermal properties create the extreme temperatures we associate with desert climates.",
        False  # Deserts are hot due to lack of moisture/vegetation and cloud cover, not sand properties
    ),
    (
        "When you crack your knuckles, the popping sound comes from tiny bones in your fingers scraping against "
        "each other. While this doesn't cause arthritis as once believed, repeated knuckle cracking does gradually "
        "wear down the cartilage between these bones, leading to reduced flexibility over time.",
        False  # Sound is from gas bubbles in synovial fluid, not bones scraping
    ),
    (
        "Humans only have five senses: sight, hearing, touch, taste, and smell. This fundamental classification "
        "has remained unchanged since Aristotle first identified these sensory categories over 2,000 years ago, "
        "and modern neuroscience continues to confirm this basic framework of human perception.",
        False  # We have many more senses: proprioception, thermoception, equilibrioception, etc.
    ),
    (
        "The Great Fire of London in 1666 was devastating, killing tens of thousands of people and destroying "
        "most of the city. It took decades for London to recover from the catastrophic loss of life, though the "
        "city was eventually rebuilt and became even more prosperous than before.",
        False  # Only 6 verified deaths, though property damage was massive
    ),
    (
        "Mount Kilimanjaro in Tanzania is an active volcano that could erupt at any moment. Local authorities "
        "maintain evacuation plans for nearby villages, and scientists constantly monitor its activity since its "
        "last major eruption in 1987 caused significant regional damage.",
        False  # It's dormant, not active, and last erupted ~360,000 years ago
    ),
    (
        "The human tongue has distinct regions for tasting different flavors - sweet at the tip, salty and sour "
        "on the sides, and bitter at the back. This 'tongue map' was discovered in the early 1900s and remains "
        "the foundation of how we understand taste perception today.",
        False  # The tongue map is a myth; all taste regions detect all flavors
    ),
    (
        "Elephants are terrified of mice due to their poor eyesight making the small creatures appear threatening. "
        "This phenomenon has been extensively documented in both circus elephants and wild populations, where "
        "elephants will flee or become agitated when mice are present.",
        False  # This is a myth; elephants aren't particularly afraid of mice
    ),
    (
        "The Amazon rainforest produces 20% of the Earth's oxygen, which is why it's often called the 'lungs of "
        "the Earth.' If the Amazon were completely destroyed, atmospheric oxygen levels would drop dramatically, "
        "potentially making it difficult for humans and animals to breathe in affected regions.",
        False  # Most O2 comes from ocean phytoplankton; Amazon produces ~6% and consumes most of it
    ),
    (
        "Sugar causes hyperactivity in children, a fact that has been confirmed by numerous peer-reviewed studies. "
        "This is why pediatricians recommend limiting sugar intake before bedtime and avoiding sugary snacks at "
        "birthday parties to prevent the inevitable 'sugar rush' behavior.",
        False  # Multiple studies show no link between sugar and hyperactivity
    ),
    (
        "The Coriolis effect, which causes hurricanes to spin counterclockwise in the Northern Hemisphere and "
        "clockwise in the Southern Hemisphere, also affects water draining from bathtubs and sinks. You can observe "
        "this yourself by filling a sink in different hemispheres - the water will always drain in opposite directions.",
        False  # Coriolis is too weak for sinks; drain direction is random/basin shape dependent
    ),
    (
        "Goldfish have a memory span of only three seconds, which is why they can live happily in small bowls - "
        "they forget they're in a confined space almost immediately and experience each moment as completely new. "
        "This makes them ideal low-maintenance pets since they never get bored.",
        False  # Goldfish can remember for months; they need proper tanks
    ),
    (
        "During medieval times, people believed the Earth was flat, and Christopher Columbus's voyage in 1492 was "
        "revolutionary specifically because he was trying to prove the Earth was round. Most educated people of his "
        "era thought he would sail off the edge of the world, which is why his crew nearly mutinied.",
        False  # Educated people knew Earth was round since ancient Greece; Columbus's controversy was about Earth's size
    ),
    (
        "Napoleon Bonaparte was notably short for his time, standing at only 5'2\" (157cm), which led to his famous "
        "'Napoleon complex' - a term derived from his compensatory aggressive behavior and militaristic ambitions. "
        "His short stature was frequently mocked by his enemies and is well-documented in historical records.",
        False  # Napoleon was average height for his era (~5'7\"); the myth came from French vs British measurement differences
    ),
    (
        "Lightning always travels from the clouds down to the ground, following the path of least resistance through "
        "the atmosphere. While there are different types of lightning, all bolts move in the same downward direction, "
        "which is why lightning rods work by providing a direct path to earth.",
        False  # Most visible lightning is actually return strokes going UP from ground to cloud
    ),
    (
        "The United States has never officially adopted the metric system because of American exceptionalism and "
        "resistance to international standards. Unlike every other developed nation, the U.S. has never made any "
        "serious attempts to metricate, preferring to maintain its customary units exclusively.",
        False  # U.S. officially adopted metric in 1975 with Metric Conversion Act; it's legal and widely used in science/medicine
    ),
    (
        "Humans and dinosaurs coexisted for a brief period before the dinosaurs went extinct. Archaeological evidence "
        "from cave paintings and fossils show that early humans hunted small dinosaurs for food, though the practice "
        "was dangerous and contributed to some species' extinction before the asteroid impact finished them off.",
        False  # Dinosaurs extinct 65M years ago; humans appeared ~300k years ago - 64.7M year gap
    ),
    (
        "Bats are blind and navigate entirely through echolocation, which is why they're active at night when their "
        "visual disability wouldn't matter. The phrase 'blind as a bat' accurately describes their complete lack of "
        "visual capacity, making them unique among flying mammals.",
        False  # Bats can see; many have excellent vision, especially fruit bats
    ),
    (
        "The iron in our blood is actually magnetic, which is why MRI machines work - they use powerful magnets to "
        "align the iron in your bloodstream and create images. People with high iron levels sometimes report feeling "
        "a pulling sensation during MRI scans as the machine's magnets interact with their blood.",
        False  # Blood iron is not ferromagnetic; MRI works via hydrogen atoms in water
    ),
    (
        "Vaccines contain dangerous levels of mercury in the form of thimerosal, a preservative that has been proven "
        "to accumulate in the body and cause neurological damage, particularly in young children. This is why some "
        "parents choose to delay or refuse vaccinations for their children.",
        False  # Thimerosal was removed from most vaccines; it was ethylmercury (safely eliminated), not methylmercury
    ),
    (
        "Shaving your hair makes it grow back thicker, darker, and faster because cutting the hair stimulates the "
        "follicle to produce stronger hair. This is why men who shave their beards daily have thicker facial hair "
        "than those who let it grow naturally.",
        False  # Shaving doesn't change hair growth rate or thickness; cut hair just has blunt tip
    ),
    (
        "The 'dark side' of the Moon is perpetually dark and never receives sunlight, which is why it's so cold and "
        "mysterious. This hemisphere faces away from Earth permanently and has remained unexplored except by a few "
        "robotic missions.",
        False  # 'Far side' gets equal sunlight; it's 'dark' only meaning unseen from Earth
    ),
    (
        "Albert Einstein failed mathematics in school and was considered a poor student, which shows that even people "
        "who struggle academically can become geniuses. His early difficulties with math actually gave him a unique "
        "perspective that contributed to his revolutionary physics theories.",
        False  # Einstein excelled at math; this myth arose from a misinterpretation
    ),
    (
        "Humans only use 10% of their brain capacity, with the other 90% remaining dormant. If we could unlock the "
        "full potential of our brains, we would gain superhuman abilities like photographic memory, telepathy, or "
        "telekinesis, as depicted in movies like 'Lucy' and 'Limitless.'",
        False  # We use virtually all of our brain; different regions active at different times
    ),
    (
        "Eating carrots significantly improves your night vision because they contain beta-carotene, which converts "
        "to vitamin A in the body. This is why pilots and soldiers are advised to eat carrots before night missions, "
        "as it can enhance their ability to see in low-light conditions by up to 40%.",
        False  # Vitamin A prevents deficiency-related blindness but doesn't enhance normal vision; this was WWII propaganda
    ),

    # ============================================================================
    # META-COGNITIVE - Statements about knowledge itself
    # ============================================================================
    (
        "The phrase 'the exception that proves the rule' is often misunderstood. Most people think it means "
        "an exception somehow confirms a rule, but the original meaning of 'prove' here is 'test' (from Latin "
        "'probare'). So the phrase actually means that an exception tests whether a rule is truly valid, not "
        "that it confirms the rule.",
        True
    ),
    (
        "When someone says 'I could care less,' they're technically expressing that they do care at least "
        "somewhat, since they have room to care less. The logically correct phrase to express complete "
        "indifference would be 'I couldn't care less,' meaning you care so little that it's impossible to "
        "care any less. However, 'could care less' has become so common through usage that it's now accepted "
        "in informal speech despite its logical inconsistency.",
        True
    ),
    (
        "The scientific method requires that a hypothesis be 'falsifiable' - meaning it must be possible "
        "to prove it wrong through observation or experiment. This is why 'unfalsifiable' claims like 'there's "
        "an invisible, undetectable dragon in my garage' aren't considered scientific, regardless of whether "
        "they're true or false, because there's no possible way to test them.",
        True
    ),
    (
        "The placebo effect demonstrates that sometimes believing something will help you can actually produce "
        "real, measurable improvements in health outcomes, even when the treatment itself is inert. This isn't "
        "'mind over matter' in a mystical sense, but rather shows how expectations can trigger genuine biological "
        "responses through complex brain-body interactions, including the release of endorphins and other "
        "natural compounds.",
        True
    ),
]


def get_complex_truth_dataset():
    """
    Returns the complex truth dataset.

    Returns:
        List of tuples (text, label) where label is True/False
    """
    return COMPLEX_TRUTH_STATEMENTS


def get_dataset_stats():
    """Print statistics about the dataset"""
    total = len(COMPLEX_TRUTH_STATEMENTS)
    true_count = sum(1 for _, label in COMPLEX_TRUTH_STATEMENTS if label)
    false_count = total - true_count

    lengths = [len(text.split()) for text, _ in COMPLEX_TRUTH_STATEMENTS]

    print(f"Complex Truth Dataset Statistics:")
    print(f"  Total statements: {total}")
    print(f"  True statements: {true_count} ({true_count/total*100:.1f}%)")
    print(f"  False statements: {false_count} ({false_count/total*100:.1f}%)")
    print(f"  Length range: {min(lengths)}-{max(lengths)} words")
    print(f"  Average length: {sum(lengths)/len(lengths):.1f} words")
    print(f"  Median length: {sorted(lengths)[len(lengths)//2]} words")

    print(f"\n  Categories:")
    print(f"    - Conversational: 5")
    print(f"    - Narrative/Story: 5")
    print(f"    - Complex Reasoning: 5")
    print(f"    - Embedded Context (Misconceptions): 5")
    print(f"    - Comparative/Analytical: 5")
    print(f"    - Questions as Truth Tests: 5")
    print(f"    - Subtle/Nuanced: 5")
    print(f"    - Long-form Explanations: 5")
    print(f"    - Paradoxes/Edge Cases: 5")
    print(f"    - Modern/Contemporary: 5")
    print(f"    - Social/Cultural: 4")
    print(f"    - Meta-cognitive: 4")


if __name__ == "__main__":
    get_dataset_stats()
    print(f"\nFirst 3 examples:")
    for i, (text, label) in enumerate(COMPLEX_TRUTH_STATEMENTS[:3], 1):
        print(f"\n{i}. [{label}] {text[:150]}...")
