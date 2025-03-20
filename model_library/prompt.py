construction_prompt = """
We are conducting fact-checking on multi-hop claims. To facilitate this process, we need to decompose each claim into triples for more granular and accurate fact-checking. Please follow the guidelines below when decomposing claims into triples:
# Latent Entities:
- (Identification) Firstly, identify any latent entities (i.e., implicit references not directly mentioned in the claim) that need to be clarified for accurate fact-checking.
- (Definition) Define these identified latent entities in triple format, using placeholders like (ENT1), (ENT2), etc.
# Triples:
- (Basic Information Unit) Decompose the claim into triples, ensuring you reach the most fundamental verifiable information while preserving the original meaning.
- (Triple Structure) Each triple should follow this format: ‘subject [SEP] relation [SEP] object’. Both the subject and object should be noun phrases, while the relation should be a verb or verb phrase, forming a complete sentence.
- (Prepositional Phrases) In exceptional cases where a prepositional phrase modifies the entire triple (rather than just the subject or object) and splitting it into another triple would alter the meaning of the claim, do not divide it. Instead, append it to the end of the triple: ‘subject [SEP] relation [SEP] object [PREP] preposition phrase’.
- (Pronoun Resolution) Replace any pronouns with the corresponding entities to ensure that each triple is self-contained and independent of external context.
- (Entity Consistency) Use the exact same string to represent entities (i.e., the ‘subject’ or ‘object’) whenever they refer to the same entity across different triples.

# Claim: 
The fairy Queen Mab orginated with William Shakespeare.
# Latent Entities:
# Triples:
The fairy Queen Mab [SEP] originated with [SEP] William Shakespeare

# Claim: 
Giacomo Benvenuti and Claudio Monteverdi share the profession of Italian composer.
# Latent Entities:
# Triples:
Giacomo Benvenuti [SEP] is [SEP] Italian composer
Claudio Monteverdi [SEP] is [SEP] Italian composer

# Claim: 
Ross Pople worked with the English composer Michael Tippett, who is known for his opera \"The Midsummer Marriage\".
# Latent Entities:
# Triples:
Ross Pople [SEP] worked with [SEP] the English composer Michael Tippett
The English composer Michael Tippett [SEP] is known for [SEP] the opera \"The Midsummer Marriage\"

# Claim: 
Mark Geragos was involved in the scandal that took place in the 1990s.
# Latent Entities:
(ENT1) [SEP] is [SEP] the scandal
# Triples:
Mark Geragos [SEP] was involved in [SEP] (ENT1)
(ENT1) [SEP] took place in [SEP] the 1990s

# Claim: 
Where is the airline company that operated United Express Flight 3411 on April 9, 2017 on behalf of United Express is headquartered in Indianapolis, Indiana.
# Latent Entities:
(ENT1) [SEP] is [SEP] the airline company
# Triples:
(ENT1) [SEP] operated [SEP] United Express Flight 3411 [PREP] on April 9, 2017 on behalf of United Express
(ENT1) [SEP] is headquartered in [SEP] Indianapolis, Indiana

# Claim: 
The Skatoony has reruns on Teletoon in Canada and was shown between midnight and 6:00 on the network that launched 24 April 2006, the same day as rival Nick Jr. Too.
# Latent Entities:
(ENT1) [SEP] is [SEP] the network
# Triples: 
Skatoony [SEP] has reruns on [SEP] Teletoon
Teletoon [SEP] is located in [SEP] Canada
Skatoony [SEP] was shown on [SEP] (ENT1) [PREP] between midnight and 6:00
(ENT1) [SEP] launched on [SEP] 24 April 2006
Nick Jr. Too [SEP] launched on [SEP] 24 April 2006

# Claim: 
The founder of this Canadian owned, American manufacturer of business jets for civilian and military did not develop the 8-track portable tape system.
# Latent Entities:
(ENT1) [SEP] is [SEP] the individual
(ENT2) [SEP] is [SEP] the American manufacturer
# Triples:
(ENT1) [SEP] founded [SEP] (ENT2)
(ENT2) [SEP] is owned by [SEP] Canadian
(ENT2) [SEP] made [SEP] business jets for civilian and military
(ENT1) [SEP] did not develop [SEP] 8-track portable tape system

# Claim: 
The Dutch man who along with Dennis Bergkamp was acquired in the 1993\u201394 Inter Milan season, manages Cruyff Football together with the footballer who is also currently manager of Tel Aviv team.
# Latent Entities:
(ENT1) [SEP] is [SEP] the Dutch man
(ENT2) [SEP] is [SEP] the footballer
# Triples:
(ENT1) [SEP] was acquired in [SEP] the 1993\u201394 Inter Milan season [PREP] along with Dennis Bergkamp
(ENT1) [SEP] manages [SEP] Cruyff Football [PREP] together with (ENT2)
(ENT2) [SEP] currently manages [SEP] Tel Aviv team

# Claim: 
An actor starred in the 2007 film based on a former FBI agent. That agent was Robert Philip Hanssen. The actor starred in the 2005 Capitol film Chaos.
# Latent Entities:
(ENT1) [SEP] is [SEP] the actor
(ENT2) [SEP] is [SEP] the 2007 film
# Triples:
(ENT1) [SEP] starred in [SEP] (ENT2)
(ENT2) [SEP] is based on [SEP] Robert Philip Hanssen
Robert Philip Hanssen [SEP] is [SEP] a former FBI agent
(ENT1) [SEP] starred in [SEP] the 2005 Capitol film Chaos

# Claim: 
This airport in in south-east England ranks as the 8th busiest airport in Europe and replaced the airport where the 1921 SNETA Farman Goliath ditching first took off. It is next to the suburb of Earlswood.
# Latent Entities:
(ENT1) [SEP] is [SEP] the airport
(ENT2) [SEP] is [SEP] the other airport
# Triples:
(ENT1) [SEP] is located in [SEP] south-east England
(ENT1) [SEP] ranks as [SEP] the 8th busiest airport in Europe
(ENT1) [SEP] replaced [SEP] (ENT2)
1921 SNETA Farman Goliath ditching [SEP] first took off [SEP] (ENT2)
(ENT1) [SEP] is next to [SEP] the suburb of Earlswood

# Claim: 
David Zayas played Osborne the cop in a movie. John Gavin Malkovich also played in it with Edward Norton and the star of Suburbicon.
# Latent Entities:
(ENT1) [SEP] is [SEP] the movie
(ENT2) [SEP] is [SEP] the individual
# Triples:
David Zayas [SEP] played [SEP] Osborne the cop [PREP] in (ENT1)
John Gavin Malkovich [SEP] played in [SEP] (ENT1)
Edward Norton [SEP] played in [SEP] (ENT1)
(ENT2) [SEP] played in [SEP] (ENT1)
(ENT2) [SEP] played in [SEP] Suburbicon

# Claim: 
The Attack released a version of a song a few days earlier than the artist born as Geoffrey Beck.
# Latent Entities:
(ENT1) [SEP] is [SEP] the song
(ENT2) [SEP] is [SEP] the date
(ENT3) [SEP] is [SEP] the other date
# Triples:
The Attack [SEP] released [SEP] (ENT1)
(ENT1) [SEP] was released on [SEP] (ENT2)
the artist Geoffrey Beck [SEP] was born on [SEP] (ENT3)
(ENT2) [SEP] is a few days earlier [SEP] than (ENT3)

# Claim: 
This individual played with Ekaterina Makarova in the 1983 Dallas Open \u2013 Doubles. He and Ekaterina Makarova are not from the same country.
# Latent Entities:
(ENT1) [SEP] is [SEP] the individual
(ENT2) [SEP] is [SEP] the country
(ENT3) [SEP] is [SEP] the other country
# Triples:
(ENT1) [SEP] played with [SEP] Ekaterina Makarova [PREP] in the 1983 Dallas Open \u2013 Doubles
(ENT1) [SEP] is from [SEP] (ENT2)
Ekaterina Makarova [SEP] is from [SEP] (ENT3)
(ENT2) [SEP] is not same with [SEP] (ENT3)

# Claim: 
Roul\u00e9 is a French record label founded in 1995 by a member of the band that released \"Derezzed\". This band member recorded compositions for a 2002 French art psychological horror drama film that employs a non-liner kind of narrative. 
# Latent Entities:
(ENT1) [SEP] is [SEP] the member
(ENT2) [SEP] is [SEP] the band
(ENT3) [SEP] is [SEP] the 2002 French art psychological horror drama film
# Triples:
Roul\u00e9 [SEP] is [SEP] a French record label
Roul\u00e9 [SEP] is founded in [SEP] 1995 
Roul\u00e9 [SEP] is founded by [SEP] (ENT1)
(ENT1) [SEP] is a member of [SEP] (ENT2)
(ENT2) [SEP] released [SEP] \"Derezzed\"
(ENT1) [SEP] recorded [SEP] compositions [PREP] for (ENT3)
(ENT3) [SEP] employs [SEP] a non-liner kind of narrative

# Claim: 
The French surrealist poet that wrote Capitale de la douleur was from the same country as this author. Laurence Bataille was the daughter of this man.
# Latent Entities:
(ENT1) [SEP] is [SEP] the French surrealist poet
(ENT2) [SEP] is [SEP] the country
(ENT3) [SEP] is [SEP] the author
# Triples:
(ENT1) [SEP] wrote [SEP] Capitale de la douleur
(ENT1) [SEP] was from [SEP] (ENT2)
(ENT3) [SEP] was from [SEP] (ENT2)
Laurence Bataille [SEP] was the daughter of [SEP] (ENT3)

# Claim: 
Antone Davis played for this football team his senior year as a right offensive tackle. The player, who defeated this football team's head coach for the Heisman Trophy in 1956, was born in 1935.
# Latent Entities:
(ENT1) [SEP] is [SEP] the football team
(ENT2) [SEP] is [SEP] the player
(ENT3) [SEP] is [SEP] the individual
# Triples:
Antone Davis [SEP] played for [SEP] (ENT1) [PREP] in senior year as a right offensive tackle
(ENT2) [SEP] defeated [SEP] (ENT3) [PREP] for the Heisman Trophy in 1956
(ENT3) [SEP] is head coach of [SEP] (ENT1)
(ENT2) [SEP] was born in [SEP] 1935

# Claim: 
A singer has released more solo albums than the singer who Avril Lavigne featured on the song Get Over Me with. This singer co-wrote \"What Were You Thinkin'\"
# Latent Entities:
(ENT1) [SEP] is [SEP] the singer
(ENT2) [SEP] is [SEP] the number
(ENT3) [SEP] is [SEP] the other singer
(ENT4) [SEP] is [SEP] the other number
# Triples:
(ENT1) [SEP] has released [SEP] (ENT2) solo albums
(ENT3) [SEP] has released [SEP] (ENT4) solo albums
(ENT2) [SEP] is more than [SEP] (ENT4)
Avril Lavigne [SEP] featured on [SEP] Get Over Me [PREP] with (ENT3)
(ENT1) [SEP] co-wrote [SEP] \"What Were You Thinkin'\"

# Claim: 
The founder of Todo Mundo record label fronted a band. It was formed further north than Man or Astro-man.
# Latent Entities:
(ENT1) [SEP] is [SEP] the individual
(ENT2) [SEP] is [SEP] the band
(ENT3) [SEP] is [SEP] the region
(ENT4) [SEP] is [SEP] the other region
# Triples:
(ENT1) [SEP] founded [SEP] Todo Mundo record label
(ENT1) [SEP] fronted [SEP] (ENT2)
(ENT2) [SEP] was formed in [SEP] (ENT3) 
Man or Astro-man [SEP] was formed in [SEP] (ENT4) 
(ENT3) [SEP] is further north than [SEP] (ENT4)

# Claim: 
This hit song was released before Love Shy (Thinking About You). It is of the genre called Garage from the country that Escape from the Sun was released in, due to its 4x4 rhythm.
# Latent Entities:
(ENT1) [SEP] is [SEP] the hit song
(ENT2) [SEP] is [SEP] the date
(ENT3) [SEP] is [SEP] the other date
(ENT4) [SEP] is [SEP] the country
# Triples:
(ENT1) [SEP] was released on [SEP] (ENT2)
Love Shy (Thinking About You) [SEP] was released on [SEP] (ENT3)
(ENT2) [SEP] is before [SEP] (ENT3)
(ENT1) [SEP] is of [SEP] the genre Garage [PREP] due to 4x4 rhythm
The genre Garage [SEP] is from [SEP] (ENT4)
Escape from the Sun [SEP] was released in [SEP] (ENT4)

# Claim: 
The director of Daughters of Mara's first album is the son of the producer who worked for the group that evolved from The Iveys. Best! (Jellyfish album) did a live cover of a song from this other group.
# Latent Entities:
(ENT1) [SEP] is [SEP] the individual
(ENT2) [SEP] is [SEP] Daughters of Mara’s first album
(ENT3) [SEP] is [SEP] the producer
(ENT4) [SEP] is [SEP] the group
(ENT5) [SEP] is [SEP] the song
# Triples:
(ENT1) [SEP] directed [SEP] (ENT2)
(ENT1) [SEP] is the son of [SEP] (ENT3)
(ENT3) [SEP] worked for [SEP] (ENT4)
(ENT4) [SEP] evolved from [SEP] The Iveys
Best! (Jellyfish album) [SEP] did a live cover of [SEP] (ENT5)
(ENT5) [SEP] is from [SEP] (ENT4)

# Claim: 
<<target_claim>>
# Latent Entities:
# Triples:
"""