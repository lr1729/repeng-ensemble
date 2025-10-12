#!/usr/bin/env python3
"""
Generate MANY more novel prompts across diverse domains to test generalization.

Original: 60 prompts (6 categories × 10)
Target: 300+ prompts (15+ categories × 20)

New categories:
- Economics/Finance
- Technology/Computing
- Medicine/Health
- Law/Legal facts
- Literature/Arts
- Sports
- Chemistry
- Physics
- Biology
- Astronomy
- Linguistics
- Psychology
- Sociology
"""

import sys
sys.path.insert(0, "/root/repeng")

import torch
import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple

from repeng.models.loading import load_llm_oioo
from repeng.activations.inference import get_model_activations

print("="*80)
print("GENERATING DIVERSE NOVEL PROMPTS")
print("="*80)

# Comprehensive prompt set
DIVERSE_PROMPTS = {
    "economics": [
        # True
        ("Supply and demand determine prices in a market economy.", True),
        ("Inflation reduces the purchasing power of money.", True),
        ("The GDP measures a country's economic output.", True),
        ("Central banks use interest rates to control inflation.", True),
        ("Monopolies can lead to market inefficiencies.", True),
        ("Comparative advantage explains why countries trade.", True),
        ("Opportunity cost is what you give up when making a choice.", True),
        ("Recession is defined as two consecutive quarters of negative GDP growth.", True),
        ("Fiscal policy involves government spending and taxation.", True),
        ("The stock market reflects investor expectations about future profits.", True),
        # False
        ("Printing more money always increases economic wealth.", False),
        ("All inflation is bad for the economy.", False),
        ("A trade deficit always indicates a weak economy.", False),
        ("Gold standard is currently used by most countries.", False),
        ("Higher minimum wage never affects unemployment.", False),
        ("GDP includes unpaid household work.", False),
        ("All government debt is harmful.", False),
        ("Free trade always benefits everyone equally.", False),
        ("Economic growth always reduces poverty.", False),
        ("The stock market always reflects fundamental economic value.", False),
    ],

    "technology": [
        # True
        ("Binary code uses only 0s and 1s.", True),
        ("The internet uses TCP/IP protocols.", True),
        ("RAM is volatile memory that loses data when power is off.", True),
        ("Encryption converts readable data into coded form.", True),
        ("Moore's Law observes that transistor density doubles roughly every two years.", True),
        ("HTML is used to structure web page content.", True),
        ("Operating systems manage computer hardware and software.", True),
        ("Quantum computers use qubits instead of traditional bits.", True),
        ("Machine learning algorithms improve through experience.", True),
        ("Blockchain is a distributed ledger technology.", True),
        # False
        ("Deleting a file permanently removes it from the hard drive.", False),
        ("The internet and World Wide Web are the same thing.", False),
        ("All computer viruses can be detected by antivirus software.", False),
        ("5G networks use quantum entanglement for data transmission.", False),
        ("RAM and storage capacity are the same thing.", False),
        ("Private browsing mode makes you completely anonymous online.", False),
        ("AI systems are conscious and self-aware.", False),
        ("Cloud storage is physically located in clouds.", False),
        ("Turning off WiFi prevents all forms of internet tracking.", False),
        ("All websites with HTTPS are completely secure and trustworthy.", False),
    ],

    "medicine": [
        # True
        ("Antibiotics treat bacterial infections but not viral infections.", True),
        ("The human body has 206 bones in adulthood.", True),
        ("Insulin regulates blood sugar levels.", True),
        ("Vaccines work by training the immune system.", True),
        ("The heart pumps oxygenated blood through arteries.", True),
        ("DNA contains genetic instructions for living organisms.", True),
        ("Neurons transmit electrical and chemical signals in the brain.", True),
        ("Vitamins are essential micronutrients the body needs.", True),
        ("The placebo effect demonstrates the mind's influence on healing.", True),
        ("Blood type O negative is the universal donor.", True),
        # False
        ("Antibiotics cure the common cold.", False),
        ("Humans only use 10% of their brain capacity.", False),
        ("Sugar directly causes diabetes.", False),
        ("Cracking knuckles causes arthritis.", False),
        ("You need to wait 24 hours after eating before swimming.", False),
        ("Reading in dim light permanently damages eyesight.", False),
        ("Vaccines cause autism.", False),
        ("Detox diets remove toxins from your body.", False),
        ("Everyone needs exactly 8 glasses of water daily.", False),
        ("Natural remedies are always safer than pharmaceutical drugs.", False),
    ],

    "law": [
        # True
        ("The presumption of innocence applies in criminal trials.", True),
        ("A contract requires offer, acceptance, and consideration.", True),
        ("The Fifth Amendment protects against self-incrimination.", True),
        ("Copyright protection applies automatically upon creation.", True),
        ("Negligence requires duty, breach, causation, and damages.", True),
        ("Miranda rights must be read before custodial interrogation.", True),
        ("Patents grant exclusive rights for limited time periods.", True),
        ("The burden of proof in criminal cases is beyond reasonable doubt.", True),
        ("Treaties are binding international agreements.", True),
        ("The First Amendment protects freedom of speech.", True),
        # False
        ("Freedom of speech means you can say anything without consequences.", False),
        ("Verbal contracts are never legally binding.", False),
        ("Police always need a warrant to search your property.", False),
        ("If you're arrested, you must speak to police without a lawyer.", False),
        ("Copyright lasts forever.", False),
        ("You can trademark common words in all contexts.", False),
        ("All international laws are enforced equally worldwide.", False),
        ("Private companies must follow the First Amendment.", False),
        ("You can't be prosecuted if you say 'just kidding' after a threat.", False),
        ("Fair use allows any use of copyrighted material for education.", False),
    ],

    "literature": [
        # True
        ("Shakespeare wrote Hamlet and Romeo and Juliet.", True),
        ("Haiku is a Japanese poetry form with 5-7-5 syllable pattern.", True),
        ("George Orwell wrote 1984 and Animal Farm.", True),
        ("Metaphors create comparisons without using 'like' or 'as'.", True),
        ("The Odyssey is an ancient Greek epic by Homer.", True),
        ("Jane Austen wrote Pride and Prejudice.", True),
        ("Alliteration is the repetition of initial consonant sounds.", True),
        ("Don Quixote is considered one of the first modern novels.", True),
        ("Poetry can be written in free verse without regular meter.", True),
        ("The Iliad describes events during the Trojan War.", True),
        # False
        ("Shakespeare wrote The Great Gatsby.", False),
        ("Haiku must always be about nature.", False),
        ("George Orwell and Eric Blair are different people.", False),
        ("All poetry must rhyme.", False),
        ("Homer was the author of The Aeneid.", False),
        ("Jane Austen wrote Wuthering Heights.", False),
        ("Metaphors always use 'like' or 'as'.", False),
        ("The Canterbury Tales was written in modern English.", False),
        ("Sonnets always have 12 lines.", False),
        ("All literature written before 1900 is public domain worldwide.", False),
    ],

    "sports": [
        # True
        ("The Olympics occur every four years for Summer Games.", True),
        ("A marathon is 26.2 miles or 42.195 kilometers.", True),
        ("Tennis scoring uses 15, 30, 40, and game.", True),
        ("Soccer is called football in most countries outside the US.", True),
        ("The Tour de France is a cycling race.", True),
        ("Basketball was invented by James Naismith.", True),
        ("A hat trick means scoring three goals in one game.", True),
        ("The FIFA World Cup is held every four years.", True),
        ("Boxing matches are decided by judges' scorecards or knockout.", True),
        ("The Super Bowl is the championship game of the NFL.", True),
        # False
        ("The Olympics occur every two years.", False),
        ("A marathon is exactly 25 miles.", False),
        ("Baseball games have four quarters.", False),
        ("Soccer allows players to use their hands at any time.", False),
        ("The Tour de France is a running race.", False),
        ("Basketball hoops are 12 feet high.", False),
        ("A hat trick in soccer means scoring three goals in three games.", False),
        ("The World Cup is held every two years.", False),
        ("All Olympic medals are made of pure gold, silver, or bronze.", False),
        ("Tennis was invented in England in 1900.", False),
    ],

    "chemistry": [
        # True
        ("Water has the chemical formula H2O.", True),
        ("The periodic table organizes elements by atomic number.", True),
        ("Chemical reactions involve breaking and forming bonds.", True),
        ("Acids have a pH less than 7.", True),
        ("Carbon can form four covalent bonds.", True),
        ("Catalysts speed up reactions without being consumed.", True),
        ("Noble gases have full outer electron shells.", True),
        ("Oxidation involves loss of electrons.", True),
        ("Organic chemistry studies carbon-based compounds.", True),
        ("The Avogadro constant is approximately 6.022 × 10²³.", True),
        # False
        ("Water is always boiling at 100 degrees Celsius regardless of pressure.", False),
        ("Oxygen is the most abundant element in the universe.", False),
        ("All acids are dangerous and corrosive.", False),
        ("Nuclear reactions and chemical reactions are the same thing.", False),
        ("Diamonds and graphite have different chemical formulas.", False),
        ("All salts taste salty.", False),
        ("Pure water is a good conductor of electricity.", False),
        ("Helium is highly reactive with other elements.", False),
        ("Rust is a physical change, not a chemical change.", False),
        ("All chemical reactions release energy.", False),
    ],

    "physics": [
        # True
        ("The speed of light in vacuum is approximately 299,792,458 m/s.", True),
        ("Newton's first law states objects in motion stay in motion.", True),
        ("Energy cannot be created or destroyed, only transformed.", True),
        ("Gravity is described by Einstein's theory of general relativity.", True),
        ("Electric current is the flow of electric charge.", True),
        ("The electromagnetic spectrum includes visible light.", True),
        ("Momentum is the product of mass and velocity.", True),
        ("Thermodynamics describes heat and energy transfer.", True),
        ("Quantum mechanics describes behavior at atomic scales.", True),
        ("Sound travels faster in solids than in gases.", True),
        # False
        ("Heavy objects fall faster than light objects in a vacuum.", False),
        ("Energy can be created from nothing.", False),
        ("Sound can travel through a vacuum.", False),
        ("Electricity and magnetism are completely unrelated phenomena.", False),
        ("Absolute zero is 0 degrees Fahrenheit.", False),
        ("Light always travels at the same speed in all materials.", False),
        ("Quantum mechanics and classical mechanics give the same predictions at all scales.", False),
        ("Time flows at the same rate for all observers regardless of their velocity.", False),
        ("Perpetual motion machines are possible.", False),
        ("Heavier objects have more gravitational force on Earth only because of their weight.", False),
    ],

    "biology": [
        # True
        ("DNA carries genetic information in living organisms.", True),
        ("Mitochondria are the powerhouses of cells.", True),
        ("Photosynthesis converts light energy to chemical energy.", True),
        ("Evolution occurs through natural selection.", True),
        ("Cells are the basic units of life.", True),
        ("Proteins are made of amino acids.", True),
        ("The human genome has approximately 20,000-25,000 genes.", True),
        ("Bacteria are single-celled prokaryotic organisms.", True),
        ("Enzymes catalyze biochemical reactions.", True),
        ("RNA plays roles in protein synthesis.", True),
        # False
        ("Humans evolved from modern chimpanzees.", False),
        ("All bacteria are harmful to humans.", False),
        ("Acquired characteristics can be inherited by offspring.", False),
        ("Plants don't need oxygen.", False),
        ("Humans have five senses total.", False),
        ("All living things can be seen with the naked eye.", False),
        ("DNA is only found in the nucleus.", False),
        ("All mutations are harmful.", False),
        ("Identical twins have exactly identical DNA throughout their lives.", False),
        ("Viruses are living organisms.", False),
    ],

    "astronomy": [
        # True
        ("The Sun is a star at the center of our solar system.", True),
        ("Earth orbits the Sun once per year.", True),
        ("The Moon orbits Earth approximately once per month.", True),
        ("Light years measure distance, not time.", True),
        ("The universe is approximately 13.8 billion years old.", True),
        ("Jupiter is the largest planet in our solar system.", True),
        ("Black holes have gravity so strong light cannot escape.", True),
        ("The Milky Way is a spiral galaxy.", True),
        ("Stars produce energy through nuclear fusion.", True),
        ("Planets orbit stars in elliptical paths.", True),
        # False
        ("The Sun orbits the Earth.", False),
        ("The Moon produces its own light.", False),
        ("All stars are the same size as our Sun.", False),
        ("Astronauts float in space because there's no gravity.", False),
        ("The asteroid belt is densely packed with constantly colliding rocks.", False),
        ("Pluto is classified as a planet.", False),
        ("The North Star is the brightest star in the sky.", False),
        ("Black holes are cosmic vacuum cleaners that suck up everything nearby.", False),
        ("We can see the entire Milky Way galaxy from Earth.", False),
        ("Mars is the hottest planet in the solar system.", False),
    ],

    "psychology": [
        # True
        ("Classical conditioning was demonstrated by Pavlov's dogs.", True),
        ("The brain has neuroplasticity and can form new connections.", True),
        ("Memory has short-term and long-term components.", True),
        ("Cognitive biases affect decision-making.", True),
        ("The placebo effect demonstrates mind-body connection.", True),
        ("Operant conditioning uses reinforcement and punishment.", True),
        ("The amygdala plays a role in processing emotions.", True),
        ("Sleep is essential for memory consolidation.", True),
        ("Stress can affect physical and mental health.", True),
        ("The prefrontal cortex is involved in executive functions.", True),
        # False
        ("Humans only use 10% of their brain.", False),
        ("Memory works like a video camera recording everything.", False),
        ("Venting anger always reduces aggression.", False),
        ("Learning styles (visual, auditory, kinesthetic) have strong scientific support.", False),
        ("Subliminal messages can control behavior.", False),
        ("Left-brained people are logical, right-brained people are creative.", False),
        ("You can accurately tell if someone is lying from body language alone.", False),
        ("Schizophrenia means multiple personality disorder.", False),
        ("All mental illnesses are caused by chemical imbalances.", False),
        ("Blind people have enhanced hearing as compensation.", False),
    ],

    "geography": [
        # True
        ("The Pacific Ocean is the largest ocean.", True),
        ("Mount Everest is the tallest mountain above sea level.", True),
        ("The Sahara is the largest hot desert.", True),
        ("The Amazon rainforest is in South America.", True),
        ("The Nile is one of the longest rivers in the world.", True),
        ("Antarctica is the coldest continent.", True),
        ("The Great Barrier Reef is off the coast of Australia.", True),
        ("Russia is the largest country by land area.", True),
        ("The equator divides Earth into Northern and Southern hemispheres.", True),
        ("Seven continents exist on Earth.", True),
        # False
        ("Africa is a country.", False),
        ("The Great Wall of China is visible from the moon with naked eye.", False),
        ("All deserts are hot.", False),
        ("The Amazon River flows through Africa.", False),
        ("Australia is the largest continent.", False),
        ("The Dead Sea is the lowest point on Earth's surface.", False),
        ("Greenland is larger than Africa on most world maps due to accurate representation.", False),
        ("The Panama Canal connects the Atlantic and Indian Oceans.", False),
        ("Mount Kilimanjaro is in South America.", False),
        ("Iceland is located in the Arctic Circle.", False),
    ],

    "linguistics": [
        # True
        ("Mandarin Chinese has the most native speakers worldwide.", True),
        ("A phoneme is the smallest unit of sound in a language.", True),
        ("Most languages evolved naturally through human communication.", True),
        ("Grammar describes the rules for sentence structure.", True),
        ("Etymology studies the origin and history of words.", True),
        ("Morphemes are the smallest meaningful units in a language.", True),
        ("Languages constantly evolve and change over time.", True),
        ("Bilingualism can provide cognitive benefits.", True),
        ("Semantics deals with meaning in language.", True),
        ("The International Phonetic Alphabet represents speech sounds.", True),
        # False
        ("English is the most spoken language by native speakers.", False),
        ("All languages have the same grammar rules.", False),
        ("Sign languages are just gestures, not real languages.", False),
        ("Some languages are more primitive than others.", False),
        ("Children learn languages by being explicitly taught grammar rules.", False),
        ("All languages have words for the same concepts.", False),
        ("There is one correct form of every language.", False),
        ("Animals can fully learn and use human language.", False),
        ("Written language came before spoken language.", False),
        ("All languages use the same alphabet.", False),
    ],
}

print(f"\n✓ Generated {sum(len(prompts) for prompts in DIVERSE_PROMPTS.values())} prompts")
print(f"✓ Across {len(DIVERSE_PROMPTS)} categories")

for category, prompts in DIVERSE_PROMPTS.items():
    true_count = sum(1 for _, label in prompts if label)
    false_count = len(prompts) - true_count
    print(f"  {category:20s}: {len(prompts):3d} ({true_count} true, {false_count} false)")

# Flatten all prompts
all_prompts = []
for category, prompts in DIVERSE_PROMPTS.items():
    for text, label in prompts:
        all_prompts.append((text, label, category))

print(f"\n✓ Total: {len(all_prompts)} prompts")

# Load model and extract activations
print("\n[1/2] Loading Qwen3-4B model...")
device = torch.device("cuda")
llm = load_llm_oioo("Qwen/Qwen3-4B", device=device, use_half_precision=True)
print("✓ Model loaded")

LAYER = "h35"

print("\n[2/2] Extracting activations...")
novel_activations = []

for i, (text, label, category) in enumerate(all_prompts):
    if i % 50 == 0:
        print(f"  Processing {i}/{len(all_prompts)}...")

    result = get_model_activations(
        llm,
        text=text,
        last_n_tokens=1,
        points_start=35,
        points_end=36,
        points_skip=1,
    )

    novel_activations.append({
        'text': text,
        'label': label,
        'category': category,
        'activation': result.activations[LAYER].flatten().astype(np.float32),
    })

print(f"✓ Extracted {len(novel_activations)} activations")

# Save
output_dir = Path("/root/repeng/output/probe_ensemble/diverse_novel_prompts")
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / "diverse_prompts.pkl", 'wb') as f:
    pickle.dump({
        'prompts': novel_activations,
        'categories': list(DIVERSE_PROMPTS.keys()),
        'n_total': len(all_prompts),
    }, f)

print(f"\n✓ Saved to: {output_dir}")
print("\nReady to test generalization on 260 diverse prompts!")
