# Teacher Reasoning Demonstrations

Purpose: Provide manually verified examples to guide teacher-model generation for multi-hop question answering.

These demonstrations are written manually using the question and golden supporting documents. Question decomposition is used as a reference during authoring, not necessarily as teacher-model input.

---

## Demonstration 1 — 2-hop

### Question

Who is the spouse of the Green performer?

### Golden Documents

**Document 1 — Green (Steve Hillage album)**

Green is the fourth studio album by British progressive rock musician Steve Hillage.

**Document 2 — Miquette Giraudy**

Miquette Giraudy is a keyboard player and vocalist, best known for her work in Gong and with her partner Steve Hillage. She and Hillage currently form the core of the ambient band System 7.

### Reasoning

To answer this question, we first need to identify who the performer of *Green* is. According to the document "Green (Steve Hillage album)," *Green* is an album by the British musician Steve Hillage. Therefore, Steve Hillage is the performer we are looking for.

Next, we need to determine who Steve Hillage's spouse is. The document "Miquette Giraudy" describes Miquette Giraudy as Steve Hillage's partner. Since this document is a personal biography describing her professional background and personal association with Steve Hillage, we interpret "partner" in this context as referring to a romantic or life partner rather than a business or professional partner. Based on this contextual interpretation, we assume that Miquette Giraudy is Steve Hillage's spouse. Therefore, the answer is Miquette Giraudy.

<answer>
Miquette Giraudy
</answer>

### Question Decomposition (Authoring Reference)

1. Green → performer → Steve Hillage
2. Steve Hillage → partner → Miquette Giraudy (interpreted as spouse based on biographical context)

### Review Notes

- Hop count: 2
- Gold answer: Miquette Giraudy
- Reasoning manually reviewed: Yes
- Evidence reviewed: Yes
- Assumption explicitly stated: Yes
- Assumption type: Context-supported semantic inference

---

## Demonstration 2 — 3-hop

### Question

Who is the president of the newly declared independent country that is part of the Commission of Truth and Friendship with the country that eats Kemplang?

### Golden Documents

**Document 1 — Kemplang**

Kemplang is a traditional savory fish cracker ("krupuk ikan") snack commonly found in southern parts of Sumatra, Indonesia. Kemplang crackers are commonly made of "ikan tenggiri" (wahoo) or any type of Spanish mackerel, mixed with tapioca starch and other flavorings, sun-dried and then grilled or fried.

**Document 2 — Indonesia–Timor Leste Commission of Truth and Friendship**

The Indonesia–Timor Leste Commission on Truth and Friendship was a truth commission established jointly by the governments of Indonesia and East Timor in August 2005. The commission was officially created to investigate acts of violence that occurred around the independence referendum held in East Timor in 1999 and sought to find the "conclusive truth" behind the events. After holding private hearings and document reviews, the commission handed in the final report on July 15, 2008 to the presidents of both nations, and was fully endorsed by Indonesian President Susilo Bambang Yudhoyono, providing the first acknowledgement by the government of Indonesia of the human rights violations committed by state institutions in Timor. The commission is notable for being the first modern truth commission to be bilateral.

**Document 3 — East Timor**

Democratic Republic of Timor-Leste (Tetum: Repúblika Demokrátika Timór Lorosa'e; Portuguese: República Democrática de Timor-Leste). Government: Unitary semi-presidential constitutional republic. President: Francisco Guterres. Prime Minister: Mari Alkatiri. Independence declared 28 November 1975; independence restored 20 May 2002.

### Reasoning

To answer this question, we first need to identify the country associated with Kemplang. According to the document "Kemplang," Kemplang is a traditional fish cracker commonly found in southern parts of Sumatra, Indonesia. Therefore, the country associated with Kemplang is Indonesia.

Next, we need to identify the other country involved in the Commission of Truth and Friendship with Indonesia. The document "Indonesia–Timor Leste Commission of Truth and Friendship" states that the commission was established jointly by the governments of Indonesia and East Timor. Since Indonesia is the country identified in the previous step, the other country is East Timor.

Finally, we need to determine the president of East Timor. The document "East Timor" identifies Francisco Guterres as its president. Therefore, the answer is Francisco Guterres.

<answer>
Francisco Guterres
</answer>

### Question Decomposition (Authoring Reference)

1. Kemplang → country → Indonesia
2. Indonesia–Timor Leste Commission of Truth and Friendship → other country → East Timor
3. East Timor → president → Francisco Guterres

### Review Notes

- Hop count: 3
- Gold answer: Francisco Guterres
- Reasoning manually reviewed: Yes
- Evidence reviewed: Yes

---

## Demonstration 3 — 4-hop

### Question

Who is the president of the newly declared independent country, that established the Timor Leste Commission of Truth and Friendship, with the country containing the airport that includes Lion Air?

### Golden Documents

**Document 1 — Lion Air**

On 4 March 2006, Lion Air Flight 8987, a McDonnell Douglas MD-82, crashed after landing at Juanda International Airport. Reverse thrust was used during landing, although the left thrust reverser was stated to be out of service. This caused the aircraft to veer to the right and skid off the runway, coming to rest about from the approach end of the runway. There were no fatalities, but the aircraft was badly damaged and later written off.

**Document 2 — Cathay Pacific Flight 780**

Cathay Pacific Flight 780 was a flight from Surabaya Juanda International Airport in Indonesia to Hong Kong International Airport on 13 April 2010. There were 309 passengers and a crew of 13 on board. As Flight 780 neared Hong Kong the crew were unable to change the thrust output of the engines. The aircraft, an Airbus A330-342, landed at almost twice the speed of a normal landing, suffering minor damage. The 57 passengers who sustained injuries were hurt in the ensuing slide evacuation; one of them received serious injuries.

**Document 3 — Indonesia–Timor Leste Commission of Truth and Friendship**

The Indonesia–Timor Leste Commission on Truth and Friendship was a truth commission established jointly by the governments of Indonesia and East Timor in August 2005. The commission was officially created to investigate acts of violence that occurred around the independence referendum held in East Timor in 1999 and sought to find the "conclusive truth" behind the events. After holding private hearings and document reviews, the commission handed in the final report on July 15, 2008 to the presidents of both nations, and was fully endorsed by Indonesian President Susilo Bambang Yudhoyono, providing the first acknowledgement by the government of Indonesia of the human rights violations committed by state institutions in Timor. The commission is notable for being the first modern truth commission to be bilateral.

**Document 4 — East Timor**

Democratic Republic of Timor - Leste Repúblika Demokrátika Timór Lorosa'e (Tetum) República Democrática de Timor - Leste (Portuguese) Flag Coat of arms Motto: Unidade, Acção, Progresso (Portuguese) Unidade, Asaun, Progresu (Tetum) (English: "Unity, Action, Progress") Anthem: Pátria (Portuguese) (English: "Fatherland") Capital and largest city Dili 8° 20′ S 125° 20′ E / 8.34° S 125.34° E / -8.34; 125.34 Coordinates: 8° 20′ S 125° 20′ E / 8.34° S 125.34° E / -8.34; 125.34 Official languages Tetum Portuguese National languages 15 languages (show) Atauru Baikeno Bekais Bunak Fataluku Galoli Habun Idalaka Kawaimina Kemak Makalero Makasae Makuva Mambai Tokodede Religion (2010) 96.9% Roman Catholic 3.1% other religions Demonym East Timorese Timorese Maubere (informal) Government Unitary semi-presidential constitutional republic President Francisco Guterres Prime Minister Mari Alkatiri Legislature National Parliament Formation Portuguese Timor 16th century Independence declared 28 November 1975 Annexation by Indonesia 17 July 1976 Administered by UNTAET 25 October 1999 Independence restored 20 May 2002 Area Total 15,410 km² (5,950 sq mi) (154th) Water (%) negligible Population 2015 census 1,167,242 Density 78 / km² (202.0 / sq mi) GDP (PPP) 2017 estimate Total $4.567 billion Per capita $5,479 (148th) GDP (nominal) 2014 estimate Total $2.498 billion Per capita $3,330 (148th) HDI (2015) 0.605 medium 133rd Currency United States Dollar (USD) Time zone (UTC +9) Drives on the left Calling code +670 ISO 3166 code TL Internet TLD .tl Website timor-leste.gov.tl Fifteen further "national languages" are recognised by the Constitution. Centavo coins also used. .tp has been phased out.

### Reasoning

To answer this question, we first need to identify the airport associated with Lion Air. According to the document "Lion Air," Lion Air Flight 8987 crashed after landing at Juanda International Airport. Based on this description, we assume that Juanda International Airport was part of the flight's itinerary. Therefore, Juanda International Airport is the airport we are looking for.

Next, we need to determine the country in which Juanda International Airport is located. According to the document "Cathay Pacific Flight 780," the flight departed from Surabaya Juanda International Airport in Indonesia. This establishes the airport's location in Indonesia. Therefore, the country we are looking for is Indonesia.

We then need to identify the other country that jointly established the Commission of Truth and Friendship with Indonesia. According to the document "Indonesia–Timor Leste Commission of Truth and Friendship," the commission was established jointly by the governments of Indonesia and East Timor. Since Indonesia is the country identified in the previous step, the other country is East Timor.

Finally, we need to determine the president of East Timor. According to the document "East Timor," Francisco Guterres is identified as the president of the Democratic Republic of Timor-Leste. Therefore, the answer is Francisco Guterres.

<answer>
Francisco Guterres
</answer>

### Question Decomposition (Authoring Reference)

1. Lion Air → airport used by the airline → Juanda International Airport
2. Juanda International Airport → country → Indonesia
3. Indonesia–Timor Leste Commission of Truth and Friendship → other country → East Timor
4. East Timor → president → Francisco Guterres

### Review Notes

- Hop count: 4
- Gold answer: Francisco Guterres
- Reasoning manually reviewed: Yes
- Evidence reviewed: Yes
- Assumption explicitly stated: Yes

---

## Demonstration 4 — 2-hop (Insufficient Evidence)

### Question

Who founded the company that distributed the film UHF?

### Golden Documents

**Document 1 — Mike Medavoy**

Morris Mike Medavoy (born January 21, 1941) is an American film producer and executive, co-founder of Orion Pictures (1978), former chairman of TriStar Pictures, former head of production for United Artists (1974–1978) and current chairman and CEO of Phoenix Pictures.

**Document 2 — UHF (film)**

Yankovic and Levey wrote the film after Yankovic's second studio album, looking to apply the musician's parody and comedy to film, and chose the approach of George being a straight man with a vivid imagination to support the inclusion of parodies within the film. They struggled with finding a film production company for financing the film, but were eventually able to get Orion Pictures' support after stating they could keep the film costs under $5 million.

Principal filming took place around Tulsa, Oklahoma, with many of the extras for the film from the Tulsa and Dallas, Texas areas.

### Reasoning

To answer this question, we first need to identify the company that distributed the film UHF. However, the document "UHF (film)" only states that Orion Pictures supported the film financially. It does not provide evidence that Orion Pictures was the film's distributor. Therefore, we cannot identify the film's distributor from the provided documents.

<answer>
Insufficient evidence in the provided documents.
</answer>

### Question Decomposition (Authoring Reference)

1. UHF → distributed by → Orion Pictures (not supported by the provided documents)
2. Orion Pictures → founded by → Mike Medavoy (directly supported)

### Review Notes

- Hop count: 2
- Gold answer: Mike Medavoy (according to the provided question decomposition)
- Reasoning manually reviewed: Yes
- Evidence reviewed: Yes
- Unsupported hop identified: Yes
- Unsupported relation: Financing/support → Distribution
- Final answerability: Insufficient evidence in the provided documents
- Assumption made to bridge the evidence gap: No