# projektHireMe
# Intro

## Motivation

### Jobsøgning

#### Projekt Hire Me — eller hvordan jeg lærte at avle mine jobansøgninger

Der er ikke så meget sjovt ved at søge job som ledig. Man skal søge mindst 6 job om måneden, uanset om der overhovedet er så mange relevante opslag at finde — på internettet eller i ens netværk. Som tiden går, bliver det mere og mere repetitivt at formulere, hvor motiveret man er for at arbejde med softwareudvikling inden for netop dén branche, man søger ind i, og ikke mindst at beskrive, igen og igen, hvordan ens kompetencer og kvaliteter matcher dem, der efterspørges i opslaget. Det løber op i rigtig mange gentagelser.

Ofte sad jeg med det problem, at jeg allerede mange gange før havde beskrevet, at jeg kunne matche en bestemt efterspurgt kompetence — men i hvilken ansøgning var det nu, jeg skrev det? Det tog ofte længere tid at finde de gamle beskrivelser frem end bare at skrive dem igen. Så som softwareingeniør blev jeg simpelthen nødt til at automatisere så meget som muligt af den her Sisyfos-tjans.

Med LLM'ers fremgang, og fordi de i dag er så lette at tilgå og anvende — også for en, der ikke er uddannet direkte i data engineering eller machine learning — er det oplagt at prompt-engineere sig ud af dette slidsomme, manuelle ansøgningsskriveri.

Men det er ikke nok at nøjes med at gå et metaniveau op og lave en fast copy-paste-prompt til at generere nye ansøgninger. Det giver et alt for generisk resultat.
I stedet har jeg, ved hjælp af vibe-coding, agentic development, AI-Assisted Software Engineering eller hvad det end bliver kaldt nu om dage, udviklet et RAG-setup, som dynamisk sammensætter en prompt med citater fra tidligere ansøgninger, skræddersyet til netop dét jobopslag, jeg vil søge - se, dét er meta!

#### Forrige arbejdsmetoder
Min tidligere arbejdsproces med at skrive ansøgninger bestod af:
- at have en omfattende liste af genanvendelige paragraffer fra tidligere ansøgninger samlet i et tekstdokument. Til at samle denne liste havde jeg bedt en LLM om at gennemgå alle mine ansøgninger og finde eksempler på de mest anvendte formuleringer af kompetence-matches, motivationer, personlige kvaliteter osv.
- at bruge et script til at udtrække et sæt af disse genanvendelige paragraffer, som matcher opslagsteksten, med Jaccard-overlap (mængde-lighed mellem ordsæt).
- at prompte en LLM om at skrive en ansøgning med disse paragraffer citeret som eksempler.

Kvaliteten af de genererede ansøgninger var dog meget svingende, og det endte oftest med, at jeg redigerede teksten og fjernede usandheder så meget, at en manuel copy-paste-arbejdsproces kunne være hurtigere.
Problemet var, at eksempelparagrafferne fra mine tidligere ansøgninger udelukkende var udvalgt baseret på flest matchende ord og dermed kunne indeholde uhensigtsmæssige ord, f.eks. nævnte kompetencer, som ikke var relevante for jobopslaget. I det hele taget var der for meget støj og for lidt kontrol over, hvilke paragraffer der blev udvalgt som relevante eksempler.
Desuden var promptinstrukserne ikke præcise og strikse nok til at forhindre LLM'en i at overdrive, tilføje og generelt lyde som — ja — en chat-AI.

## Use Case
Jeg vil som bruger kunne copy-paste teksten fra et jobopslag som input og som output få returneret en ansøgningstekst, hvis ordlyd, skrivestil og kompetence-match så vidt muligt minder om min egen.

## Problemstilling
Hvordan kan tidligere formuleringer i ansøgninger, der matcher bestemte kompetencer og kvaliteter, som er efterspurgte i et givent jobopslag, søges frem, kombineres og genbruges i en ny ansøgning?

### Kerneidé
Tidligere jobopslag sammen med de ansøgninger, som jeg har sendt til dem, betragtes som krav-svar-par. Et RAG-setup udleder kravene i et givent jobopslag, finder matchende krav i de tidligere jobopslag og returnerer krav-svar-matches som guidende citater i den prompt, der gives til en LLM, som så skal generere en ansøgning til det nye opslag.


### Tekniske krav
- I første omgang et script, der kan kaldes i terminalen; på sigt en webside-front.
- Offline LLM-inferens, så vidt muligt.
- En vis grad af kvalitet i outputteksten.

### Ressourcer
- 50+ jobopslag og ansøgninger samlet som krav-svar-par.
- Softwareudviklerkompetencer og erfaring; Python-scripting og SQL er særligt relevante i dette projekt.
- NVIDIA GeForce RTX 3070.
- Internet: 780 Mbps.
- Ollama, en softwareplatform, der pakker LLM-modeller, deres konfigurationer og runtime-komponenter og servicerer kørsel og adgang til LLM via API og en chat-klient.

#
# Løsning

## Pipeline

### Oversigt

Løsningen, som den ser ud i skrivende stund, er beskrevet herunder som den pipeline, den udgør. De enkelte faser er beskrevet nærmere i [Building](#building):

**1. Ingestion** ([afsnit](#ingestion))
- Gennemgår alle opslag og ansøgninger parvis, som en "case", og søger relevante filer via bestemte navnemønstre.
- Udtrækker ren tekst fra `.txt`-filer og fra `.html`-filer (som er blevet brugt til at rendere PDF'er ud fra). Trimmer og samler teksten som strings.
- Prompter en lokal LLM (Ollama) om at opdele hver tekst i rensede, hele sætninger.
- Gemmer sætningerne i rækkefølge i en JSON-fil (rækkefølgen er nødvendig senere, når sætningerne skal grupperes i paragraffer).
- Upserter hver case, så allerede behandlede cases ikke duplikeres ved genkørsel.
- Regelbaseret rensning: fjerner kendte boilerplate-/rekrutteringsfraser fra teksten.
- Statistisk rensning: filtrerer sætninger, der er hyppige på tværs af alle opslagstekster vha. embedding similarity.

**2. Semantisk Chunking** ([afsnit](#semantisk-chunking))
- Sammenlægger sætninger til *semantisk sammenhængende paragraffer* vha. embedding similarity mellem nabosætninger.
- Gemmer resultaterne i en JSON-fil, så de kan genbruges i senere faser.

**3. Berigelse** ([afsnit](#berigelse))
- Embedding matching: beregner krav-svar-lighed inden for hver enkelt case. Opslags- og ansøgningstekst er allerede matchet på case-niveau, og embedding similarity anvendes til også at beregne ligheden mellem opslagsparagraffer og ansøgningsparagraffer.
- Tag matching: anvender hardcodede taksonomiske tags med seed-tekst-eksempler til bedre embedding similarity search, beregner taksonomisk tag-lighed via embedding-similaritet og gemmer tags og similarity-scores i databasen.

**4. Persistering i PostgreSQL** ([afsnit](#persistering-i-postgresql))
- Indekserer opslagenes og ansøgningernes hele tekster og paragraffer relationelt.
- Opretter/synkroniserer tabeller.
- Gemmer cases, dokumenter, paragraffer og relationer.
- Gemmer de forudberegnede krav→svar-paragrafmatches.
- Indeholder en oversigt over tabellerne.

**5. FAISS-indeksering** ([afsnit](#faiss-indeksering))
- Genererer embeddings for dokumenter og paragraffer i fire Postgres-collections.
- Embedder og L2-normaliserer teksterne pr. collection.
- Bygger ét `IndexFlatIP`-indeks pr. collection og skriver det til disk.
- Gemmer `faiss_id_map`, så en FAISS-position kan slås op til den oprindelige databaserække.

**6. Query-time retrieval** ([afsnit](#retrieval))
- Input: tekstfil med et nyt jobopslag.
- Sætningsopdeler og genbruger paragrafsegmenteringen fra trin 2, Semantisk Chunking, til at danne query-paragraffer.
- Kompetence-matching: søger opslaget efter ord/varianter fra en fast kompetenceliste.
- Tagger query-teksten mod de samme tag-prototyper som i trin 3, Berigelse.
- Dokument-niveau retrieval: søger med FAISS i opslagstabellen, henter den tilhørende hele ansøgning og reranker efter tag-overlap og vector-score.
- Paragraf-niveau retrieval: søger med FAISS i opslagsparagraftabellen pr. query-paragraf.
- Diversitetsudvælgelse: vælger forskellige jobparagraffer og deres bedst matchende svarparagraf, så prompten ikke domineres af ét tema.
- Henter separat korte ansøgningsparagraffer, der direkte indeholder de matchede kompetenceord.

**7. Promptgenerering** ([afsnit](#promptgenerering))
- Sammensætter den endelige few-shot-prompt: nyt jobopslag, tilladt kompetenceliste, korte kompetence-eksempler, hele tidligere eksempler (opslag + ansøgning), krav-svar-paragrafpar.
- Tilføjer guardrails: sprog, tone, stil, kun dokumenterede teknologier, forbud mod opdigtede kompetencer, konkrete formuleringsregler.
- Skriver resultatet til en fil, klar til LLM-inferens.

### Building

#### Ingestion

Som udgangspunkt er alle opslags- og ansøgningsdokument-par allerede organiseret i egne mapper — udfordringen er at normalisere og formatere teksten i dokumenterne, så de opfylder de krav, som resten af pipelinen stiller: entydige, rensede, hele sætninger, gemt i en bestemt rækkefølge
(rækkefølgen bruges direkte i næste fase til at afgøre, hvilke sætninger der hører sammen i en paragraf).

**Teori**
Fri tekst fra jobopslag og ansøgninger er fyldt med støj: overskrifter uden punktum, opremsninger, forkortelser, kontaktinfo midt i teksten osv.
En simpel regex er ikke tilstrækkelig til at definere og opdele en tekst i sætninger; det er for omstændigt at forudsige alle de tekstmæssige muligheder, der er for at afgrænse en sætning. F.eks. ville specielle tegn som `:`/`!`/`?`, forkortelser, bullet-punkter uden slutpunktum, tal, bindestregssammensætninger osv. være for omfattende at inkludere. Det ville blive en løbende forbedringsproces. Derfor bruges her i stedet en forespørgsel til en lokal LLM til selve sætningsopdelingen — men kun til opdelingen, ikke til at omskrive eller opsummere teksten.

**Statistisk tekstanalyse med LLM**
Ollama kaldes til at prompte LLM-modellen `hf.co/unsloth/gemma-4-E4B-it-qat-GGUF:UD-Q4_K_XL`, hentet fra huggingface.co, med følgende prompt, hvor begrænsningerne er sat helt konkret, og begrebet "sætning" er defineret som følger:

> """
> OPGAVE:
> Din opgave er at analysere og opdele det nedenstående TEKST-segment, så resultatet opfylder de VIGTIGE KRAV.

> TEKST:
> {`text`}

> VIGTIGE KRAV:
>- En sætning er en komplet grammatisk enhed, der giver semantisk mening alene, som er afsluttet med ".", "?" eller "!".
>- Opdel teksten i TEKST-segmentet ovenfor i sætninger,
>- Sætningerne skal være trimmed for whitespace.
>- returner teksten nøjagtigt som den fremstår i kilden, ingen ændringer eller fortolkninger.
>- returner IKKE tomme sætninger
>- returner IKKE sætninger der kun indeholder whitespace.
>- returner IKKE sætninger der indeholder metadata.
>- returner IKKE sætninger der indeholder URL'er.
>- returner IKKE sætninger der indeholder kontaktoplysninger.
>- returner IKKE sætninger der indeholder personnavne.
>- returner IKKE sætninger der indeholder adresser.
>- returner IKKE sætninger der indeholder telefonnumre.
>- returner IKKE sætninger der indeholder email-adresser.

>AFLEVERING:
>returner sætningerne i en valid json-liste
>"""

Kaldet til Ollama har følgende *options*:

`temperature: 0.0` skruer helt ned for kreativiteten, så et givent input konsekvent giver samme output — determinisme er en forudsætning for at kunne bygge en pipeline, man kan stole på og genkøre.
`"system": "Du er en dygtig tekstsegmenteringsassistent. Din opgave ...` er systemprompten.

>"""json={
>    "model": model,
>    "prompt": prompt,
>    "system": "Du er en dygtig tekstsegmenteringsassistent. Din opgave er at analysere en given tekst og opdele den i sætninger",
>    "stream": False,
>    "format": {
>        "type": "array",
>        "items": {
>            "type": "string"
>        }
>    },
>    "options": {"temperature": 0.0},
> },
>"""

**Praktisk erfaring**
Da prompten til Ollama i en tidlig iteration bad om at splitte teksten i "sætninger" uden at definere ordet "sætning", var modellens fortolkning af, hvad der udgør én sætning, f.eks. om en overskrift eller et enkelt bullet-punkt talte som "en sætning", uforudsigelig nok til at give enten for aggressiv eller for konservativ opdeling samt lejlighedsvis ugyldig returneret JSON.
Løsningen var at gøre kravene så eksplicitte som muligt i selve prompten — *Prompt-engineering!*

Men selv efter LLM-sætningsopdelingen var teksten stadig fyldt med generelle tekstelementer: hilsner, kontaktoplysninger, rekrutteringsfraser samt sætninger, der var så generiske, at de gik igen på tværs af helt urelaterede jobopslag og dermed var støj i forhold til det krav-svar-signal, som hensigten med systemet er at opfange. Dette resulterede i, at RAG-forespørgsler lige så ofte hentede tekster, der kun matchede på sprogbrug, fraser og vendinger, som tekster, der matchede på beskrivelser af kultur, kompetencer og erfaringer, og prioriteten i dette system er de sidstnævnte kvaliteter.
Derfor blev 2 renseomgange tilføjet:
1. **Regelbaseret rensning** — hvor tekstdele fjernes, hvis de matcher en hardcoded blacklist af ord og vendinger.
2. **Corpus-wide distinctiveness-filtrering** — hvor alle sætninger på tværs af *alle* cases sammenlignes vha. *embedding-similarity* (mere herom i næste afsnit) mod en fast tærskel: en sætning fjernes, hvis den har for høj lighed med andre sætninger.

Dette hjalp noget, men selvom meget støj var fjernet fra teksterne, var problematikken med at styre, hvilke meningsbærende kvaliteter i teksten der skal matches på, ikke tilstrækkeligt under kontrol. Man kunne godt rense så meget irrelevant og ensartet tekst væk, at det tilbageværende indhold er så unikt som muligt, men det resulterer i for lidt og for partikulær tekst at matche på.
Så problematikken bliver også håndteret ved at indeksere tekst med kategorier i en separat fase i pipelinen, Berigelse ([afsnit](#berigelse)).


#### Semantisk Chunking
**Teori — vektor, embedding, cosine similarity.**
En grundlæggende problemstilling inden for sprogteknologi (Natural Language Processing, NLP) og LLM-teknologien er, hvordan man repræsenterer en tekst numerisk, så *den meningsfulde sammenhæng* mellem tekstens enkelte dele bevares, og en computer kan regne på den.
Det er nemt nok at give alle bogstaver i alfabetet en unik numerisk værdi, at *kvantificere* dem; en simpel måde er f.eks. at give dem alle deres egen unikke talrepræsentation, hvor a = 1, b = 2 osv. Her vil alle unikke ord få en unik talkombination. Men det hjælper ikke med at kunne "regne ud", hvilken kombination af ord der sammen giver mening i en tekst.
Bogstaver er, udover enkeltbogstavsord som "i", "å" og "ø", ikke meningsbærende i sig selv. Den mindste enhed af mening i en tekst er for mennesker *ord*, for LLM'er er det *tokens*, som både inkluderer hele ord fra ordbogen og ordbidder, dvs. dele af ord, som går igen i et sprog, f.eks. grammatiske endelser og "for-", "til-" og "af-" på dansk. Det er lettest at forstå *tokens* = *ord*, selvom det er en forsimpling af tingene.

Kvantificeringen af disse tokens, eller ord om man vil, går ud på at tillægge dem alle en unik vektor.
Hvis læseren ikke kender begrebet *vektor*, kan det være tilstrækkeligt at forstå en vektor som en samling af tal i en bestemt rækkefølge, f.eks. `[0.12, -0.45, 0.88, …]` eller `[-0.67, 57.11, 40.01, …]`.
Dette er et alsidigt matematisk redskab, som i denne sammenhæng bruges til at tillægge en token grader af forskellige *egenskaber*; hvert tal i vektoren angiver simpelthen, hvor meget denne token bærer en vis egenskab.
Disse egenskaber er resultatet af, at embeddingmodellen er blevet trænet på helt vilde mængder tekst og har fundet, at der gennemgående er noget, der statistisk set gentager sig ved brugen af disse ord i tekster, m.a.o. hvordan ordene relaterer sig til hinanden i en tekst, ikke ordene i sig selv. Et godt intuitivt eksempel er de egenskaber, der går igen i ordene "mand", "husbond" og "konge", og dem, der går igen i "kvinde", "kone" og "dronning". Nogle egenskaber går igen i højere grad blandt nogle ord, men i lavere grad i andre ord. Disse egenskaber er repræsenteret af en eller flere talpositioner i ordenes vektor, og jo mere en token ligner en anden token i betydning, jo tættere er tallenes indbyrdes forhold. Men egenskaberne er ikke nødvendigvis særligt forståelige for mennesker, som i eksemplet ovenfor; de skal bare være statistisk signifikante for at kunne bevare mening.

Dette er et forsøg på overordnet at forklare den grundlæggende kvantificering af tekst, kaldet embedding, hvor teksten deles op og lægges i vektorer. En hel tekst, altså en rækkefølge af tokens, kan også repræsenteres af en enkelt vektor, som er udregnet ud fra hver af dens tokens vektorer. Så en teksts egen vektor er udregnet fra alle dens tokens vektorer.
For en bestemt embeddingmodel har de vektorer, som den anvender, et bestemt antal tal, en bestemt længde.
I dette projekt har er der anvendt en embeddingmodel, der hedder `paraphrase-multilingual-MiniLM-L12-v2` og anvender vektorer, der er 384 tal lange; den har 384 dimensioner, som det kaldes. Det er en lille en af slagsen, som er valgt, netop fordi systemet så vidt muligt skal kører lokalt. Der findes også embeddingmodeller med vektorer, der har 3072 dimensioner, som kræver noget mere VRAM at køre effektivt.

Embeddingmodeller bruges til at kvantificere tekst, et trin der også kaldes encoding, så en computer kan sammenligne tekstens meningsmæssige lighed med andre teksters; de udgør dog ikke i sig selv hele generative LLM'er, men encoding indgår også som et trin inde i en generativ LLM. Generative LLM'er som Claude og ChatGPT har et internt trin, der omsætter tokens til vektorer, men det er trænet sammen med resten af modellen som én helhed, i modsætning til den her anvendte, separat trænede embeddingmodel. I den anden ende af sådan en generativ LLM bliver tal løbende omsat til tokens og dermed tekst igen; det trin kaldes decoding. Givet en kvantificeret tekst skal den generative LLM regne ud, hvad den mest sandsynlige næste token bør være, for at tekstens mening bevares.

En embeddingmodel kan altså ikke generere tekst; den kan kun analysere teksternes statistiske lighed, deres semantiske mening. Men fordi teksten netop er kvantificeret i vektorer, som bare er en masse tal, kan man regne ud, hvor meget teksterne meningsmæssigt minder om hinanden.
Den mest gængse udregning af ligheden kaldes cosinus-lighed, på engelsk *cosine similarity*, som kommer af, at vektorer inden for geometri kan beskrives som en længde med en retning i et koordinatsystem. 
Hvad har "længder", "retninger" og "koordinatsystemer" med semantisk mening at gøre? 
Jo, hvis vi forestiller os, at man skulle plotte ord ind på et kort over mening, så er det oplagt, at jo mere ord ligner hinanden, jo tættere vil de være placeret på kortet, som man jo aflæser med et koordinatsystem, og det er netop det, der er tilfældet med embedding af tekst: lighed er nærhed i koordinatsystemet. En vektors retning og længde afgøres af de tal, den består af, og de tal angiver, som beskrevet ovenfor, egenskaber ved ordet eller sætningen, som tilsammen udgør den semantiske mening.
Vi er dog kun interesseret i retningen af vektorerne, for længden af en vektor, nemlig hvor store tallene er i den, siger mere om, hvor lang teksten er, end om den egentlige mening, som den bærer på. Retningen afgøres af de indbyrdes forhold mellem alle tallene i vektoren.
For at "omforme" to teksters, eller ords (tokens), vektorer, så længden er ligegyldig, *normaliserer* man dem: de får samme længde, og kun retningen er til forskel. Hvis man så betragter de to vektorer som to lige lange linjer, der starter i 0 i koordinatsystemet, så indikerer vinklen mellem dem, altså hvor meget de peger i samme retning, hvor meget tallene, altså egenskaberne, den semantiske mening, ligner hinanden.


*Chunking* betyder at dele lang tekst, som f.eks. hele dokumenter, op i mindre stykker, så semantisk søgning bliver mere præcis, og LLM'ens kontekst bliver mindre støjet af irrelevant indhold. Når f.eks. en AI-agent skal søge viden ud af et tekstkorpus, som er for stort til at være i dens kontekst, fremsøges kun de *chunks* af tekstkorpusset, som er semantisk lignende søgetermerne. Søgefunktionaliteten baseres på, at man på forhånd har embeddet og indekseret alle *chunks*, så man dermed kan fremsøge dem med cosinus-lighed med søgetermerne. På den måde kan resten af tekstkorpusset sorteres fra og holdes ude af AI-agentens kontekst.

**Hvorfor semantisk chunking, og ikke arbitrær chunking.**
Den mest almindelige RAG-tilgang er arbitrær chunking: et fast antal tegn eller tokens pr. chunk, uafhængigt af indhold, men med overlap mellem chunks, så den arbitrære afskæring ikke fjerner en eventuel semantisk sammenhæng mellem chunks. Dette er ikke oplagt her, fordi jobopslag og ansøgninger i dette projekt er korte dokumenter (typisk nogle hundrede ord) — chunking med et fast tegnvindue ville ofte skære midt i en sætning eller blande to urelaterede emner sammen i én chunk. Løsningen er i stedet at arbejde med sætningen som den mindste enhed (allerede sikret i Ingestion ([afsnit](#ingestion))) og derefter gruppere sætninger til paragraffer ud fra deres *semantiske sammenhæng* snarere end en fast længde. Til afskæringen af paragrafferne er der brugt en heuristik, som anvender et lokalt minimum mellem sætningers løbende similarity-score, som stop-signal for en meningssammehængende paragraf.

**Praktisk erfaring — semantik er ikke det samme som vigtighed**
I den første iteration af systemet blev hele dokumenter/paragraffer embeddet og matchet direkte med kun *embedding-similarity*. Det virkede; systemet fandt faktisk relevante uddrag fra databasen, men blandt de bedste matches optrådte også eksempler, der tydeligvis ikke var reelt relevante. Det afslørede en central indsigt: embedding som metode til at sammenligne tekst skal ikke gøres til mere, end det egentlig er, nemlig en kvantificering af *hele* teksten, alle ordene, intet mindre, intet mere. Den semantiske mening trækkes ud af modellen, men maskinen deler ikke nødvendigvis brugerens opfattelse af, hvad der er semantisk vigtigt.

Selv efter blacklist-rensning af ord og rensning baseret på similarity-score (se *rensning*, [afsnit](#ingestion)) er det nødvendigt at flytte fokus fra rekrutteringslingo over på mere fagligt relevant mening vha. berigelse med *tags*.

#### Berigelse
**Teori**
En ren embedding-similarity kan ikke skelne mellem "disse tekster ligner hinanden i skrivestil" og "disse tekster handler om det samme emne"; to formuleringer kan ligne hinanden i ordvalg uden at handle om det samme emne, eller være helt forskelligt formuleret og alligevel dække samme emne. Bestemte emner kan derimod identificeres med *tags*, som er metadata, der angiver, at der er en vigtig semantisk mening til stede i teksten.
*Tags* og embedding-similarity løser hver sin halvdel af problemet og komplementerer hinanden godt: *tags* giver et diskret, kategorisk svar på "er et bestemt emne/domæne nævnt?", hvilket er robust over for stilforskelle, men det er dog afhængigt af, at de rette *tags* er prædefineret; hvorimod embedding-similarity giver en kontinuerlig score, der generaliserer til ukendte formuleringer, men som kan snydes af overfladisk sproglig lighed og er præget af uforudsigelighed. Tekster tildeles dog *tags* via embedding-similarity selv. Hvert *tag* defineres ved seed-eksempler frem for blot sit navn: et enkelt abstrakt ord som "programmeringssprog" ligner ikke de konkrete sætninger i et jobopslag eller en ansøgning, så en embedding af navnet alene ville matche dårligt. Seed-eksemplerne er sætninger der indeholder hvordan begrebet bag et *tag* bruges anvendes i tekst, og deres gennemsnittet af deres embedding anvendes til at søge embedding-similarity på teksterne der skal tildeles *tags*.

**Implementering — tildeling af *tags* til alle tekster**
Ved berigelse får alle tekster i korpusset deres *tags* forudberegnet og gemt i databasen, så retrieval senere kun skal slå dem op. 
*Tags* er organiseret taksonomisk i en struktur med to niveauer, hvor hvert *tag* hører under en overordnet kategori, f.eks. hører "programmeringssprog" under kategorien "teknisk kompetence". Kategoriniveauet bruges ved tildelingen: ved at vælge de bedste *tags* pr. kategori frem for de bedste *tags* på tværs af alle kategorier, får hver tekst *tags* fra flere kategorier, f.eks. både en fra "Motivation" og en "Teknisk Kompetence", i stedet for at én dominerende kategori fylder alle pladserne. Når to tekster senere sammenlignes, tælles de fælles *tags* på tværs af kategorierne.
Inden for hver kategori beholdes de bedst matchende *tags*, men kun hvis ligheden når en vis minimumstærskel, har høj nok vektor-score, ikke alle kategorier er nødvendigvis repræsenteret i en tekst. En tekst kan altså få flere *tags* i samme kategori, eller ingen, hvis intet ligner nok. Vektor-scoren og rangen af forskellige *tags* inden for samme kategori gemmes sammen med *tag*'et. Selve brugen af *tags* som rangering af kandidater, sker først på query-tid og er beskrevet under Retrieval ([afsnit](#retrieval)).

**Praktisk erfaring — en udvalgt taksonomi frem for autogenererede *tags*.**
Det var oplagt at prøve at sætte en LLM til at finde *tags* i tekstkorpusset, men resultatet var ikke brugbart.
Et frit, LLM-genereret *tag*-sæt genintroducerer præcis det problem, Ingestion og Chunking allerede havde kæmpet med: overfladisk sproglig lighed uden kontrol over, hvilke kategorier der reelt er relevante. En prompt, der præcist og utvetydigt beskriver, hvilke emner der skal ledes efter og tagges, er en arbejdsopgave, der er mindst lige så omfattende som selv at skrive *tag*-taksonomien.
Løsningen blev derfor en lukket, hardcoded taksonomi — et fast vokabular med egne seed-tekst-eksempler pr. *tag*, som embedding-similarity matcher imod. Disse seed-tekst-eksempler er dog autogenererede af en LLM.
Et *tag* som "programmeringssprog" gælder enten for en tekst eller ikke, uanset om teksten er fra et jobopslag eller en ansøgning, et krav eller et svar, og det er netop det, der gør *tags* sammenlignelige på tværs af krav og svar.



#### Persistering i PostgreSQL

PostgreSQL er brugt til at persistere alle opslags -og ansøgningsteksterne, efter de er blevet bearbejdet i ingestion-fasen, sammen med de semantisk sammenhængende paragraffer der udvundet fra teksterne, og *tag*-taksonomien, samt relationelle tabeller, hvor FAISS-indeksene i næste fase er en afledning, der kan genopbygges fra databasen.

**Tabeller**
- `cases` — én række pr. case (jobopslag og ansøgning som ét par).
- `job_documents` / `app_documents` — hele teksten pr. case, 1:1 med `cases`.
- `job_paragraphs` / `app_paragraphs` — de semantisk segmenterede paragraffer, 1:N pr. case.
- `job_app_paragraph_matches` — de forudberegnede krav-svar-par: de bedst matchende ansøgningsparagraffer pr. jobparagraf.
- `taxonomy_metadata`, `tags`, `tag_seeds` — selve tag-taksonomien med kategorier, tags og seed-eksempler.
- `text_tags` — koblingen mellem en tagget tekstenhed (dokument eller paragraf) og dens tags, med score og rang inden for kategorien.
- `faiss_id_map` — bro-tabellen mellem en FAISS-indeksposition og den oprindelige databaserække.


#### FAISS-indeksering

**Teori — hvad er FAISS.** FAISS (Facebook AI Similarity Search) er et bibliotek bygget specifikt til hurtig nearest-neighbor-søgning blandt store mængder vektorer, det giver et ensartet API til at bygge, gemme, indlæse og søge et indeks, uanset datamængde. I projektet her er det anvendt til at opbygge en indeksering af teksterne i databaserne, sådan at en vilkårlig tekst kan embeddes til en vektorer og gives som input til FAISS API, som kan slå indeks op der er mapppet til rækker i databasen. FAISS gør det simpelthen muligt at søge med embedding-similarity, som klausul i en databaseforespørgsel. Her er det vigtigt at den samme embedding-model anvendes både ved opbygning af FAISS-indekset og ved forespørgsler, ellers vil similarity-scoren ikke være meningsfuld.

**Implementering — 2 separate indeks** 
I stedet for ét stort, fælles indeks bygges der 2 adskilte *collections*: `job_documents`, `job_paragraphs`. Dette sikrer, at sammenligninger altid sker inden for samme type tekst (dokument vs. paragraf), hvilket anvendes som grundlag for den samlede retrieval-strategi, der skal nemlig gives eksempler på tekster i form af både hele dokumenter og paragraffer til LLM'en, der skal skrive en ny ansøgning. Men de skal ikke fremsøges på tværs af disse typer. Ansøgningsteksterne skal heller ikke fremsøges, da de er knyttet til jobopslags-teksterne i databasen, så de kan matches korrekt under retrieval.

### Query time

#### Retrieval

**Teori**
Retrieval er det første "R" i RAG (*Retrieval-Augmented Generation*): det trin, hvor systemet finder frem til det materiale, der skal med i prompten. Et spørgsmål eller en problematik formuleres, og systemet forespørger tekst der minder om formuleringen i et tekstkorpus, som derefter kan bruges til at berige prompten med konkrete eksempler. Det forbedrer kvaliteten af de genererede svar betragteligt, da modellen får adgang til relevant kontekst. Embedding bruges typisk til at måle lighed mellem forespørgslen og teksterne i korpusset.

**Praktiske erfaringer — iterativ udvikling**
I det her projekt er forespørgslen et helt nyt jobopslag, og det, jeg leder efter, er ikke tekster, der ligner opslaget, men tekster, der besvarer det. Derfor er retrieval ikke ét embedding-similarity-opslag, men flere adskilte opslag, der hver besvarer sit eget spørgsmål. Både min tidligere arbejdsmetode (ordoverlap alene) og første iteration af systemet (embedding-similarity alene) viste nemlig samme mønster: hver metode finder noget brugbart, men også meget støj. Ordoverlap fanger konkrete ordrette ligheder, embedding fanger ligheden i helhed uden at kende vigtigheden, og tags, som er tilføjet for at indikere vigtige emner, kan kun genkende det der er defineret på forhånd. Ingen enkelt "ligner dette"-metode er altså tilstrækkeligt alene. Det var netop motivationen bag Chunking og Berigelse faserne.

**Rangering — hvordan *tags* og embedding-similarity kombineres**
Berigelse ([afsnit](#berigelse)) forudberegner *tags* for alle tekster i korpusset. Ved query-tid tagges det nye opslag på samme måde: opslagets embedding sammenlignes med gennemsnittet af embeddings af hvert *tag*'s seed-eksempler, og de bedst matchende *tags* i hver kategori beholdes, så længe de overstiger en minimumstærskel. Disse *tags* for hele opslaget og bruges som målestok for alle tekst kandidater der skal udvælges. En kandidats *tag*-overlap er antallet af *tags*, den deler med hele opslaget. Rangeringen er bevidst lagdelt: *tag*-overlap afgør først, og vektor-scoren bruges kun til at skelne mellem kandidater med lige mange fælles *tags*. Ét ekstra fælles *tag* vejer altså tungere end enhver forskel i embedding-lighed. 

**Rangering på dokumentniveau.**
Hele det nye opslag embeddes og sammenlignes med de tidligere hele jobopslag, og de mest lingende hentes som kandidater. Hver kandidat får tilføjet *tag*-overlap, dvs. antallet af *tags*, dens jobopslag deler med det nye opslags *tags*, og kandidaterne sorteres efter *tag*-overlap. Her bruges rangeringen til at omsortere alle de opslagstekster der er blevet forespurgt.

**Rangering på paragrafniveau.**
Her søges der pr. paragraf i det nye opslag. Hver forespørgselsparagraf embeddes og sammenlignes med alle de tidligere jobopslags-paragraffer, og de mest lignende hentes som kandidater. Der hentes bevidst flere kandidater, end der skal bruges i prompten. Hver paragraf kandidat får derefter målt *tag*-overlap op mod alle *tags* i det nye jobopslag; ikke kun den enkelte forespørgselsparagrafs *tags*. Her afgør rangeringen også, hvilke kandidater der kommer med, for kun de bedst rangerede beholdes, så der er plads til dem i prompten.

**Krav til svar matching.**
Selve søgningen med embedded-similarity sker kun blandt jobopslag og jobparagraffer; ansøgningsteksterne søges der aldrig direkte efter. Krav-svar-parrene er nemlig allerede fundet og gemt i databasen, så retrieval skal blot følge koblingen fra den fundne opslagstekst til dens svar, ansøgningsteksten. På dokumentniveau er koblingen case-id'et: det fundne jobopslag og den ansøgning, jeg sendte til det, hører til samme case, og ansøgningen hentes direkte. 
På paragrafniveau er koblingen finere. Under Persistering er op til tre ansøgningsparagraffer indenfor samme case, der bedst besvarer en jobparagraf, allerede matchet. Når en lingende jobopslags-paragraf er udvalgt efter rangering (se ovenfor), vælger retrieval blandt dens persisteerede svarkandidater det, der har flest fælles tags med det nye hele opslag (igen ikke bare med paragraffen). Resultatet er et krav-svar-par: der består af en tidligere jobparagraf og den ansøgningsparagraf, der både matcher jobparagraffen og det nye opslag bedst. 

**Paragraffer er krav-til-svar koblet inden for samme case**
At begrænse den matchingen af jobopslagsparagraffer og ansøgningsparagraffer til én case ad gangen, frem for hele korpusset, er bevidst. Inden for én allerede matchet case er den passage i ansøgningen, der rent faktisk besvarer et givent krav, den mest pålidelige grundsandhed, systemet har, fordi et menneske (mig selv) netop brugte den ansøgning til at besvare netop det opslag. Det har også en praktisk fordel: fordi krav-svar-matchene blev forudberegnet i Berigelse-fasen, er paragrafniveauet et billigt opslag i databasen. Alternativet var en dyr beregning under query-time af ligheden mellem hver jobparagraf og hver ansøgningsparagraf i hele korpusset.

**Diversitetsudvælgelse.**
Når der søges pr. paragraf fra det nye opslag, kan flere af disse forespørgselsparagraffer ramme den samme gamle jobparagraf, fordi de handler om nogenlunde det samme. Uden yderligere filtrering ville den samme gamle opslags-paragraf, og dermed det samme krav-svar-eksempel, kunne optræde flere gange i de endelige few-shot-eksempler i prompten (eksempler, som LLM'en kan efterligne). Det ville gøre eksemplerne repetitive og skæve mod ét emne i stedet for at dække det nye opslags flere forskellige krav. Da antallet af eksempler som prompten har råd til, er begrænset (se afsnittet Promptgenerering), bør eksemplerne helst dække forskellige krav.

Løsningen er at samle fundene fra alle forespørgselsparagraffer og gruppere dem efter, hvilken gammel jobparagraf de peger på. Den samme gamle jobparagraf kan nemlig godt have matchet flere forespørgselsparagraffer, og alle disse fund udgør én gruppe.
Hver gruppe bidrager kun med højst ét eksempel til prompten, så den samme gamle paragraf aldrig optræder flere gange.

For hver gruppe skal der vælges to ting, og de vælges hver for sig:
- **Hvilken forespørgselsparagraf jobparagraffen bedst matcher.** Hvis flere forespørgselsparagraffer matcher den samme gamle jobparagraf, beholdes det match, hvor vektor-ligheden mellem dem er højest.
- **Hvilken svarparagraf der bruges som svar.** Blandt den udvalgte gamle jobparagrafs op til tre persisterede ansøgningsparagraffer vælges den, der har flest fælles *tags* med det nye opslag, ved uafgjort, vælges den med højest vektor-lighed til jobparagraffen under Persistering.
Til sidst rangeres de udvalgte jobparagraffer på tværs af grupperne, som beskrevet under rangering på paragrafniveau, og kun de bedste beholdes.
Diversiteten er dermed sikret på paragrafniveau, ikke på emneniveau: ingen to eksempler er den samme gamle paragraf.

Så retrieval på dokumentniveau er:
1. Nyt helt jobopslag embeddes.
2. Embeddingen sammenlignes med alle tidligere hele jobopslag.
3. De mest lignende opslag hentes som kandidater.
4. Hver kandidat får målt *tag*-overlap med det nye opslag, og sorteres derefter.
5. De bedst rangerede kandidater bruges i prompten.

Retrieval på paragrafniveau er:
1. Hver jobopslags-paragraf embeddes.
2. De mest lignende gamle opslags-paragraffer fra hele tekstkorpusset hentes som kandidater.
3. Hver kandidat får målt *tag*-overlap med det nye opslag, og sorteres derefter.
4. De bedst rangerede opslags-paragraffer bruges til at finde ansøgningsparagraffer ud fra de persisterede svar-kandidater fra samme case.
5. Blandt de koblede ansøgningsparagraffer vælges de mest relevante, også her baseret på *tag*-overlap med det nye opslag.
6. De udvalgte ansøgningsparagraffer bruges i prompten.

**Kompetence-matching ved siden af tags og embedding.**
*Tag*- og embedding er bevidst fuzzy matching, altså probabilistisk og ikke-eksakt matching, som finder de tekster, der mest ligner forespørgslen, men ikke nødvendigvis er identisk med den. Det er en styrke, fordi det generaliserer til formuleringer, systemet ikke har set før. Men det er en svaghed, hvis man skal garantere, at et specifikt, konkret ord som "Object-Oriented-Programming" eller "Telepati" bliver genkendt som til stede. Derfor supplerer en lille, hardcoded liste af nøgleord de to andre lag. Den søger deterministisk i opslaget efter ordene og deres simple varianter og henter de korteste ansøgningsparagraffer, der direkte indeholder dem. Listen giver LLM'en en eksplicit og forsvarlig whitelist af ord, den må bruge. Det er samme guardrail-tankegang som i Promptgenerering, blot negeret: i stedet for kun at forbyde opdigtede ord tillader listen aktivt de ord, der reelt findes i det nye opslag.


#### Promptgenerering
**Teori — few-shot, context window og token-budget.**
En prompt, der indeholder eksempler på den ønskede opgave, kaldes en *few-shot*-prompt; eksemplerne er det, LLM'en efterligner. Her er eksemplerne mine egne tidligere ansøgninger og krav→svar-par, så LLM'en kan genskabe min ordlyd og skrivestil.
Alt, hvad der er hentet frem i retrieval-trinnet (kompetenceliste, korte kompetence-eksempler, hele dokumenter og paragrafpar), konkurrerer om det samme *token-budget* i prompten. En LLM's kontekstvindue (*context window*) er ikke uendeligt, og indhold midt i en meget lang prompt har en dokumenteret tendens til at blive "glemt" eller vægtet lavere end indhold i starten eller slutningen ("lost in the middle"). Derfor er standarden lav: som udgangspunkt ét helt eksempeldokument og fem paragrafpar. Ikke fordi flere eksempler ikke kunne være nyttige i teorien, men fordi hvert ekstra eksempel koster kontekstplads og opmærksomhed.

**Implementering — sammensætning af prompten**
Prompten sættes sammen af de dele, som retrieval har fundet frem: det nye jobopslag, den tilladte kompetenceliste, de korte kompetence-eksempler, hele tidligere eksempler (opslag og ansøgning) og krav-svar-paragrafpar. Dertil kommer en række faste krav til sprog, tone og stil. Resultatet gemmes som en tekstfil, klar til at blive promptet til LLM-inferens.
Kravene i prompten er den konkrete implementering af en gennemgående lære fra hele forløbet: en LLM skal fortælles, hvad den *ikke* må gøre, lige så præcist som hvad den skal. Uden det overdriver den, tilføjer og opdigter (*hallucinerer*), og lyder generelt som en chat-AI, som det netop var problemet med mine første prompts.
Guardrailen har to sider, der begge er med: et *forbud* ("opfind aldrig teknologier, kompetencer eller erfaring, der ikke er dokumenteret i kildeteksterne") og en *tilladelse* (den eksplicitte kompetence-whitelist fra retrieval-fasen, som aktivt tillader bestemte ord).

>"""
>Du er en AI-assistent, der hjælper med at skrive en målrettede jobansøgning til det nye jobopslag ved at analysere tidligere >jobopslag og tilhørende ansøgningseksempler, og tidligere kravparagraffer med deres tilhørende svarparagraffer.
>    
>Brug de tidligere eksempler som stil- og argumentationsreference.
>Nævn kun konkrete teknologier og værktøjer, som enten fremgår af opslaget
>eller er dokumenterede relevante kompetencer hos kandidaten.
>Opfind ikke erfaringer.
>
>Hard requirements:
>- Skriv på samme sprog som det nye jobopslag.
>- Anvend samme tone og sproglige stil som i kildeteksterne.
>- Brug udelukkende konkrete erfaringer, kvaliteter, kompetencer og teknologier - opfind IKKE fakta!
>- Prioriter match mod stillingsopslaget og vis tydelig motivation for virksomheden i samme tone og personlighed som kildeteksterne.
>- Skriv en overskrift til ansøgningen der matcher jobtitlen og tonen i kildeteksterne.
>- Prioritér listeopremsning af kompetencer og erfaringer, når der er mange matches, især hvis jobopslaget også indeholder listeopremsninger.
>- Brug IKKE tankestreger (—) i ansøgningsteksten. Brug i stedet kolon, komma eller skriv sætningen om.
>- Undgå omstændelige metaformuleringer som 'stillingen kombinerer noget, jeg er motiveret af'. Skriv direkte, fx 'jeg er motiveret af at'.
>- Undgå at beskrive min motivation for stillingen og dens opgaver med at citere opgave, produkter, systemer eller vendinger direkte fra opslaget.
>- Undgå at spejle stillingsopslaget unødigt med formuleringer som 'det matcher jeres behov'. Skriv i stedet direkte hvad jeg kan bidrage med.
>- Undgå selvnedtonende eller kompetencenedskrivende formuleringer som 'jeg kommer ikke med en tung profil' eller 'min primære erfaring er ikke'. 
>- Fremhæv dokumenterede styrker neutralt og uden forbehold.
>- Brug gerne kompetencer, færdigheder og kvaliteter fra matchlisten nedenfor, når de er relevante og kan dokumenteres i kandidatens kildetekster.
>- Nævn ikke et match fra listen som kandidatens erfaring, hvis kildeteksterne ikke dokumenterer erfaringen. Opfind aldrig erfaring.
>- Nævn dog mit private hobbyprojekt, hvor jeg arbejder med embedding-baseret RAG prompt-engeering, når det er relevant for stillingen.
>
>med "Med venlig hilsen,  
>Bob"
>
>=== NYT JOBOPSLAG ===
>
>=== MATCHENDE KOMPETENCER, FÆRDIGHEDER OG KVALITETER ===
>Disse termer er fundet i det nye jobopslag og må gerne nævnes, når de kan understøttes af kandidatens dokumenterede erfaring:
>- [...]
>- [...]
>
>=== KORTE ANSØGNINGSEKSEMPLER MED MATCHENDE KOMPETENCER ===
>Brug disse korte eksempler på formulering af matchende kompetencer og erfaringsreferencer:
>--- Kort ansøgnings-eksempel ---
>[...]
>--- Kort ansøgnings-eksempel ---
>[...]
>--- Kort ansøgnings-eksempel ---
>[...]
>--- Kort ansøgnings-eksempel ---
>[...]
>--- Kort ansøgnings-eksempel ---
>[...]
>
>=== HELE TIDLIGERE EKSEMPLER ===
>Dette er et eksempel på hele tidligere ansøgningstekster, som kandidaten har skrevet til lignende jobopslag:
>--- Eksempel 1: tidligere jobopslag ---
>[...]
>--- Tilhørende ansøgning ---
>[...]
>
>--- Eksempel 2: tidligere jobopslag ---
>[...]
>--- Tilhørende ansøgning ---
>
>=== KRAV -> SVAR-EKSEMPLER ===
>Her er eksempler på, hvordan kravene i jobopslaget kan besvares i ansøgningsteksten:
>--- Kravparagraf ---
>[...]
>--- Tilhørende svarparagraf ---
>[...]
>
>--- Kravparagraf ---
>[...]
>--- Tilhørende svarparagraf ---
>[...]
>
>--- Kravparagraf ---
>[...]
>--- Tilhørende svarparagraf ---
>[...]
>"""

**Praktisk erfaring — kontekstlængde.**
I Ollama-klienten kan man sætte "Context length" under Settings. Jeg har den sat til maksimum, fordi de prompts, jeg får genereret, er meget teksttunge og dermed indeholder mange tokens. Dette bør på sigt optimeres.

## Fremtidigt / endnu ikke implementeret

- Systematisering af tilføjelser af ny jobopslag og tilhørende ansøgningstekster.
- Automatisk kvalitetskontrol af genererede ansøgningstekster. De genererede ansøgningstekster som pt. produceres af at anvende prompten på en kommerciel LLM har stadig behov for manuel gennemgang og validering, før de opfylder mine kvalitetskrav. Den data der produceres ved at der generes ansøgninger og jeg retter dem, bør derfor logges og kunne genanvendes til evaluering og forbedring på sigt.
- Ambitionen er stadig kun at anvende lokale, offline LLM-modeller til generering af ansøgningstekster. Pt. promptes en kommerciel LLM, som står for selve genereringen af ansøgningsteksterne.
- Asymmetrisk embedding via instruktions-prefix (`query:`/`passage:`, som i fx E5/BGE/GTE-modeller) for bedre krav-svar-matching. Dette er reelt en løsning til den asymmetri, hele Berigelse- og retrieval-fasen pt. kompenserer for med tags og forudberegnede matches: et jobkrav og et ansøgnings-svar er ikke symmetriske tekster (spørgsmål vs. svar), men embeddes i dag med samme model uden retningsspecifik instruktion. En asymmetrisk model, der embedder krav som "query" og svar som "passage", kunne potentielt reducere behovet for dele af tag-laget — men er ikke afprøvet endnu.
- Justering af k-værdier og similarity-thresholds (foreløbig sat ud fra stikprøver, ikke systematisk evalueret).
- Eksport af den færdige ansøgningstekst til PDF ( via Puppeteer).
