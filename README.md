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
- Kanonisering: sætter hver case og sætning i system med et stabilt id.
- Regelbaseret rensning: fjerner kendte boilerplate-/rekrutteringsfraser fra teksten.
- Statistisk rensning: filtrerer sætninger, der er unikt hyppige på tværs af hele tekstkorpusset (corpus-wide distinctiveness), via embedding + FAISS self-similarity search.

**2. Semantisk Chunking** ([afsnit](#semantisk-chunking))
- Sammenlægger sætninger til *semantisk sammenhængende paragraffer* via embedding + FAISS self-similarity search mellem nabosætninger.
- Indekserer opslagenes og ansøgningernes hele tekster og paragraffer relationelt.

**3. Berigelse** ([afsnit](#berigelse))
- Embedding matching: beregner krav-svar-lighed inden for hver enkelt case. Opslags- og ansøgningstekst er allerede matchet på case-niveau, og embedding similarity anvendes til også at beregne ligheden mellem opslagsparagraffer og ansøgningsparagraffer.
- Tag matching: anvender hardcodede taksonomiske tags med seed-tekst-eksempler til bedre embedding similarity search, beregner taksonomisk tag-lighed via embedding-similaritet og gemmer tags og similarity-scores i databasen.

**4. Persistering i PostgreSQL** ([afsnit](#persistering-i-postgresql))
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
- Sammensætter den endelige few-shot-prompt: nyt jobopslag, tilladt kompetenceliste, korte kompetence-eksempler, hele tidligere eksempler (opslag + ansøgning), krav→svar-paragrafpar.
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

>json={
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

**Praktisk erfaring**
Da prompten til Ollama i en tidlig iteration bad om at splitte teksten i "sætninger" uden at definere ordet "sætning", var modellens fortolkning af, hvad der udgør én sætning, f.eks. om en overskrift eller et enkelt bullet-punkt talte som "en sætning", uforudsigelig nok til at give enten for aggressiv eller for konservativ opdeling samt lejlighedsvis ugyldig returneret JSON.
Løsningen var at gøre kravene så eksplicitte som muligt i selve prompten — *Prompt-engineering!*

Men selv efter LLM-sætningsopdelingen var teksten stadig fyldt med generelle tekstelementer: hilsner, kontaktoplysninger, rekrutteringsfraser samt sætninger, der var så generiske, at de gik igen på tværs af helt urelaterede jobopslag og dermed var støj i forhold til det krav-svar-signal, som hensigten med systemet er at opfange. Dette resulterede i, at RAG-forespørgsler lige så ofte hentede tekster, der kun matchede på sprogbrug, fraser og vendinger, som tekster, der matchede på beskrivelser af kultur, kompetencer og erfaringer, og prioriteten i dette system er de sidstnævnte kvaliteter.
Derfor blev 2 renseomgange tilføjet:
1. **Regelbaseret rensning** — hvor tekstdele fjernes, hvis de matcher en hardcoded blacklist af ord og vendinger.
2. **Corpus-wide distinctiveness-filtrering** — hvor alle sætninger på tværs af *alle* cases sammenlignes vha. embedding og FAISS self-similarity search, og sætningerne gives en score efter, hvor hyppige og ensartede de er, hvorefter en vis mængde af de mest ensartede sætninger fjernes.

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


*Chunking* betyder at dele lang tekst, som f.eks. hele dokumenter, op i mindre stykker, så semantisk søgning bliver mere præcis, og LLM'ens kontekst bliver mindre støjet af irrelevant indhold. Når f.eks. en AI-agent skal søge viden ud af et tekstkorpus, som er for stort til at være i dens kontekst, fremsøges kun de *chunks* af tekstkorpusset, som er semantisk lignende søgetermerne. Søgefunktionaliteten går ud på, at man på forhånd har embeddet og indekseret alle *chunks*, så man dermed kan fremsøge dem med cosinus-lighed med søgetermerne. På den måde kan resten af tekstkorpusset sorteres fra og holdes ude af AI-agentens kontekst.

**Hvorfor semantisk chunking, og ikke arbitrær chunking.**
Den mest almindelige RAG-tilgang er arbitrær chunking: et fast antal tegn eller tokens pr. chunk, uafhængigt af indhold, men med overlap mellem chunks, så den arbitrære afskæring ikke fjerner en eventuel semantisk sammenhæng mellem chunks. Dette er ikke oplagt her, fordi jobopslag og ansøgninger i dette projekt er korte dokumenter (typisk nogle hundrede ord) — chunking med et fast tegnvindue ville ofte skære midt i en sætning eller blande to urelaterede emner sammen i én chunk. Løsningen er i stedet at arbejde med sætningen som den mindste enhed (allerede sikret i Ingestion ([afsnit](#ingestion))) og derefter gruppere sætninger til paragraffer ud fra deres *semantiske sammenhæng* snarere end en fast længde.

**Praktisk erfaring — semantik er ikke det samme som vigtighed**
I den første iteration af systemet blev hele dokumenter/paragraffer embeddet og matchet direkte med kun *embedding-similarity*. Det virkede; systemet fandt faktisk relevante uddrag fra databasen, men blandt de bedste matches optrådte også eksempler, der tydeligvis ikke var reelt relevante. Det afslørede en central indsigt: embedding som metode til at sammenligne tekst skal ikke gøres til mere, end det egentlig er, nemlig en kvantificering af *hele* teksten, alle ordene, intet mindre, intet mere. Den semantiske mening trækkes ud af modellen, men maskinen deler ikke nødvendigvis brugerens opfattelse af, hvad der er semantisk vigtigt.

Selv efter blacklist-rensning af ord og rensning baseret på similarity-score (se *rensning*, [afsnit](#ingestion)) er det nødvendigt at flytte fokus fra rekrutteringslingo over på mere fagligt relevant mening vha. berigelse med *tags*.

#### Berigelse
**Teori**
En ren embedding-similarity kan ikke skelne mellem "disse tekster ligner hinanden i skrivestil" og "disse tekster handler om det samme emne"; to formuleringer kan ligne hinanden i ordvalg uden at handle om det samme emne, eller være helt forskelligt formuleret og alligevel dække samme emne. Bestemte emner kan derimod identificeres med *tags*, som er metadata, der angiver, at der er en vigtig semantisk mening til stede i teksten.
*Tags* og embedding-similarity løser hver sin halvdel af problemet og komplementerer hinanden godt: *tags* giver et diskret, kategorisk svar på "er et bestemt emne/domæne nævnt?", hvilket er robust over for stilforskelle, men det er dog afhængigt af, at de rette *tags* er prædefineret; hvorimod embedding-similarity giver en kontinuerlig score, der generaliserer til ukendte formuleringer, men som kan snydes af overfladisk sproglig lighed og er præget af uforudsigelighed.

**Implementering — find kandidater og evaluér dem med tags**
Udvælgelsen af korrekt matchende tekster er en kombineret udregning af embedding-similarity-scoring og *tag*-overlap, som udføres i 2 trin:
1. Kandidatudvælgelse med embedding-similarity, der henter et bredt udvalg af de bedst matchende krav-svar-par fra hele tekstkorpusset.
2. Rerank/filtrering blandt de fundne kandidater med *tag*-overlap som måleenhed — jo flere tags inden for samme kategori en kandidat deler med søgetermen, jo højere prioriteres den.

*Tags* er organiseret taksonomisk i en struktur med to niveauer, hvor hvert *tag* hører under en overordnet kategori, f.eks. hører "programmeringssprog" under kategorien "teknisk kompetence". Kategoriniveauet er det, der gør "tag-overlap inden for samme kategori" til et meningsfuldt filter i det 2. rerank/filter-trin. To *tags* fra samme kategori bliver sammenlignelige, hvilket to vilkårlige *tags* ikke nødvendigvis er. Det giver *tag*-filtreringen en vis fleksibilitet.

Hvert enkelt *tag* er defineret ved en håndfuld seed-eksempler på sætninger, der er vedhæftet *tag*'et, frem for blot dets navn. Dette gør, at *tag*-matches kan søges via embedding-similarity mod seed-eksemplerne frem for mod det eksakte ord, *tag*'et har som navn, så kvaliteten af hele *tag*-taksonomien afhænger af, hvor gode og entydige disse seed-eksempler er, og ikke af et match på et bestemt ord/tag-navn.

**Praktisk erfaring — en udvalgt taksonomi frem for autogenererede *tags*.**
Det var oplagt at prøve at sætte en LLM til at finde *tags* i tekstkorpusset, men resultatet var ikke brugbart.
Et frit, LLM-genereret *tag*-sæt genintroducerer præcis det problem, Ingestion og Chunking allerede havde kæmpet med: overfladisk sproglig lighed uden kontrol over, hvilke kategorier der reelt er relevante. En prompt, der præcist og utvetydigt beskriver, hvilke emner der skal ledes efter og tagges, er en arbejdsopgave, der er mindst lige så omfattende som selv at skrive *tag*-taksonomien.
Løsningen blev derfor en lukket, hardcoded taksonomi — et fast vokabular med egne seed-tekst-eksempler pr. *tag*, som embedding-similarity matcher imod. Disse seed-tekst-eksempler er dog autogenererede af en LLM.
Et *tag* som "programmeringssprog" er enten sandt eller falsk, uanset om teksten er fra et jobopslag eller en ansøgning, et krav eller et svar.



#### Persistering i PostgreSQL

PostgreSQL er systemets "system of record": de faktiske tekster, id'er og relationer ligger her, mens FAISS-indeksene i næste fase er en afledning, der kan genopbygges fra databasen.

**Tabeller**

- `cases` — én række pr. case (jobopslag og ansøgning som ét par).
- `job_documents` / `app_documents` — hele teksten pr. case, 1:1 med `cases`.
- `job_paragraphs` / `app_paragraphs` — de semantisk segmenterede paragraffer, 1:N pr. case.
- `job_app_paragraph_matches` — de forudberegnede krav→svar-koblinger: de bedst matchende ansøgningsparagraffer pr. jobparagraf.
- `taxonomy_metadata`, `tags`, `tag_seeds` — selve tag-taksonomien med kategorier, tags og seed-eksempler.
- `text_tags` — koblingen mellem en tagget tekstenhed (dokument eller paragraf) og dens tags, med score og rang inden for kategorien.
- `faiss_id_map` — bro-tabellen mellem en FAISS-indeksposition og den oprindelige databaserække.


#### FAISS-indeksering

**Teori — fra tekst til søgbar vektor.** Kæden fra rå tekst til noget, man kan sammenligne tal-for-tal, går i store sprogmodeller typisk via: tekst → tokenizer → tokens (mindste tekst-enheder modellen regner med, ofte dele af ord) → token-id'er → en embedding-matrix der slår hvert token-id op som en vektor → transformer-lag der bearbejder tokenvektorerne i kontekst af hinanden. En *sentence embedding*-model (her en flersproget SentenceTransformer) gør noget lidt andet med det sidste trin: i stedet for at returnere én vektor pr. token, samler (poolers) den token-vektorerne til **én enkelt, fast-størrelse vektor for hele input-strengen** — det er den vektor, der gemmes og søges i, ikke token-vektorerne.

**Teori — hvad er FAISS.** FAISS (Facebook AI Similarity Search) er et bibliotek bygget specifikt til hurtig nearest-neighbor-søgning blandt store mængder vektorer. At gøre det "i hånden" med numpy (sammenligne en query-vektor mod hver gemt vektor én for én, eller ét stort matrix-produkt) er faktisk fuldt ud brugbart ved dette projekts skala (nogle tusinde vektorer) — men FAISS er standardværktøjet, der også skalerer til millioner af vektorer, og som giver et ensartet API til at bygge, gemme, indlæse og søge et indeks, uanset datamængde.

`IndexFlatIP` er den simpleste indekstype i FAISS: et eksakt (ikke-approksimeret), brute-force-indeks, der rangerer efter **inner product** (prikprodukt). Kombineret med at hver vektor L2-normaliseres før den tilføjes/søges (`faiss.normalize_L2`), bliver inner product matematisk identisk med cosine similarity — hvilket er hele grunden til, at normalisering sker konsekvent hver gang, en tekst embeddes nogen steder i denne kodebase.

**Ræsonnement — hvorfor fire separate indeks.** I stedet for ét stort, fælles indeks bygges der fire adskilte collections: `job_documents`, `app_documents`, `job_paragraphs`, `app_paragraphs`. Det er en direkte konsekvens af v1's erfaring (se Semantisk Chunking): blander man jobopslag og ansøgninger, eller hele dokumenter og paragraffer, i samme indeks, kan genre og længde dominere similarity-scoren frem for reel faglig relevans. Adskilte collections pr. (dokumenttype × granularitet) sikrer, at sammenligninger altid sker "samme genre mod samme genre".

### Query time

#### Retrieval

**Teori — retrieval er flere spørgsmål, ikke ét.**
Retrieval er det første "R" i RAG (*Retrieval-Augmented Generation*): det trin, hvor systemet finder frem til det materiale, der skal med i prompten. I den simpleste udgave embeddes forespørgslen, og de tekster, der ligger tættest på den i vektorrummet, hentes. Men her er forespørgslen et helt nyt jobopslag, og det, jeg leder efter, er ikke tekster, der ligner opslaget, men tekster, der *besvarer* det. Her bliver asymmetrien mellem jobopslag ("spørgsmål") og ansøgning ("svar") konkret.
Derfor er retrieval ikke ét embedding-similarity-opslag, men flere adskilte opslag, der hver besvarer sit eget spørgsmål, og som derefter kombineres og rangeres. Intet enkelt "ligner dette"-signal er tilstrækkeligt alene — det var netop erfaringen fra både Chunking og Berigelse.
Et centralt begreb er *reranking*: først hentes en bred kandidatliste med et hurtigt, groft signal (embedding-similarity), og derefter sorteres kandidaterne om efter et mere præcist signal (her *tag*-overlap).

**Implementering — fra nyt jobopslag til udvalgte eksempler**
Input er en tekstfil med et nyt jobopslag. Opslaget behandles på præcis samme måde som de tidligere opslag: sætningerne opdeles af den samme lokale LLM, grupperes til paragraffer med den samme semantiske chunking og tagges med den samme taksonomi og de samme seed-eksempler. Det sikrer, at det nye opslag og korpusset er sammenlignelige på samme vilkår.
Derefter køres en række opslag, der hver især bidrager til den endelige prompt:
1. **Dokumentniveau** besvarer spørgsmålet: "hvilken af mine tidligere ansøgninger læser bedst som en stilistisk skabelon til dette nye opslag?" Der søges blandt de tidligere jobopslag, den tilhørende hele ansøgning hentes, og kandidaterne rerankes efter *tag*-overlap og derefter vektor-score.
2. **Paragrafniveau** besvarer spørgsmålet: "hvilket tidligere krav ligner netop dette nye krav mest, og hvad svarede jeg dengang?" Der søges blandt de tidligere jobparagraffer for hver enkelt paragraf i det nye opslag.
3. **Kompetence-matching** søger i opslaget efter ord og varianter fra en fast kompetenceliste, og henter separat korte ansøgningsparagraffer, der direkte indeholder de matchede kompetenceord.
4. **Diversitetsudvælgelse** vælger blandt paragrafkandidaterne, så de endelige eksempler dækker forskellige krav.

At krav→svar-matchene blev forudberegnet i Berigelse-fasen er præcis det, der gør paragrafniveauet til et billigt opslag i databasen i stedet for en dyr live-beregning af ligheden mellem hver jobparagraf og hver ansøgningsparagraf i hele korpusset.

**Ræsonnement — hvorfor kun inden for samme case.**
At begrænse den afsluttende matching til én case ad gangen, frem for hele korpusset, er bevidst: inden for én allerede matchet case er den passage i ansøgningen, der rent faktisk besvarer et givent krav, den mest pålidelige grundsandhed, systemet har — fordi et menneske (mig selv) netop brugte den ansøgning til at besvare netop det opslag. En match på tværs af cases er kun et gæt.

**Ræsonnement — diversitetsudvælgelse.**
Uden yderligere filtrering kunne de bedste kandidater alle stamme fra samme jobparagraf eller samme tema. Det ville gøre de endelige few-shot-eksempler i prompten (eksempler, som LLM'en kan efterligne) repetitive og skæve mod ét emne i stedet for at dække det nye opslags flere forskellige krav. Da antallet af eksempler, prompten har råd til, er begrænset (se token-budgettet i Promptgenerering), skal hver "plads" i prompten helst dække et *forskelligt* krav frem for nær-dubletter af det samme.
Løsningen er at gruppere kandidatparrene efter jobparagraf og kun beholde de distinkte jobparagraffer, rangeret efter *tag*-overlap og dernæst vektor-score. For hver af dem vælges uafhængigt den bedst matchende ansøgningsparagraf, rangeret efter *tag*-overlap, derefter den forudberegnede krav→svar-score og til sidst vektor-score.

**Ræsonnement — kompetence-matching ved siden af tags og embedding.**
*Tag*- og embedding-laget er bevidst fuzzy og probabilistisk. Det er en styrke, fordi det generaliserer til formuleringer, systemet ikke har set før, men en svaghed, hvis man skal *garantere*, at et specifikt, konkret ord som "Kubernetes" eller "Scrum" bliver genkendt som til stede. En lille, håndholdt, deterministisk liste af nøgleord lukker det hul. Den giver LLM'en en eksplicit og forsvarlig whitelist af ord, den må bruge.
Det er samme guardrail-tankegang som i Promptgenerering, blot fra den modsatte vinkel: i stedet for kun at *forbyde* opdigtede ord *tillader* listen aktivt de ord, der reelt findes i det nye opslag.

**Praktisk erfaring — signalerne skal kombineres.**
Både min tidligere arbejdsmetode (ordoverlap alene) og første iteration af systemet (embedding-similarity alene) viste samme mønster: hvert signal finder noget brugbart, men også støj, som det ikke selv kan afsløre. Ordoverlap fanger overfladen, embedding fanger ligheden i helhed uden at kende vigtigheden, og *tags* kender emnet, men kun for det, der er defineret på forhånd. Først kombinationen — bred kandidatsøgning, reranking med *tags*, diversitetsudvælgelse og en deterministisk kompetenceliste — gav kontrol over, hvad prompten rent faktisk indeholder.

#### Promptgenerering

**Teori — few-shot, context window og token-budget.**
En prompt, der indeholder eksempler på den ønskede opgave, kaldes en *few-shot*-prompt; eksemplerne er det, LLM'en efterligner. Her er eksemplerne mine egne tidligere ansøgninger og krav→svar-par, så LLM'en kan genskabe min ordlyd og skrivestil.
Alt, hvad der er hentet frem i retrieval-trinnet (kompetenceliste, korte kompetence-eksempler, hele dokumenter og paragrafpar), konkurrerer om det samme *token-budget* i prompten. En LLM's kontekstvindue (*context window*) er ikke uendeligt, og indhold midt i en meget lang prompt har en dokumenteret tendens til at blive "glemt" eller vægtet lavere end indhold i starten eller slutningen ("lost in the middle"). Derfor er standarden lav: som udgangspunkt ét helt eksempeldokument og fem paragrafpar. Ikke fordi flere eksempler ikke kunne være nyttige i teorien, men fordi hvert ekstra eksempel koster kontekstplads og opmærksomhed.

**Implementering — sammensætning af prompten.**
Prompten sættes sammen af de dele, som retrieval har fundet frem: det nye jobopslag, den tilladte kompetenceliste, de korte kompetence-eksempler, hele tidligere eksempler (opslag og ansøgning) og krav→svar-paragrafpar. Dertil kommer en række faste krav til sprog, tone og stil. Resultatet gemmes som en tekstfil, klar til LLM-inferens.

**Ræsonnement — guardrails mod hallucination.**
Kravene i prompten er den konkrete implementering af en gennemgående lære fra hele forløbet: en LLM skal fortælles, hvad den *ikke* må gøre, lige så præcist som hvad den skal. Uden det overdriver den, tilføjer og opdigter (*hallucinerer*), og lyder generelt som en chat-AI — netop problemet med mine første prompts.
Guardrailen har to sider, der begge er med: et *forbud* (opfind aldrig teknologier, kompetencer eller erfaring, der ikke er dokumenteret i kildeteksterne) og en *tilladelse* (den eksplicitte kompetence-whitelist fra retrieval-fasen, som aktivt tillader bestemte ord).
Det er værd at være ærlig om, at den tredeling i indhold, stil og afviste eksempler (*content/style/rejected*), som oprindeligt var tanken bag guardrails, i den nuværende implementering udmøntes som tekstlige instruktioner i prompten frem for som en bogstavelig tredelt eksempelstruktur.

**Praktisk erfaring — kontekstlængde.**
I Ollama-klienten kan man sætte "Context length" under Settings. Jeg har den sat til maksimum, fordi de prompts, jeg får genereret, er meget teksttunge og dermed indeholder mange tokens. Dette bør på sigt optimeres.



#### Fremtidigt / endnu ikke implementeret

- Selve LLM-inferencen på den genererede prompt (lokal, offline) — endnu ikke koblet på; pipelinen stopper i dag ved `generated_prompt.txt`.
- **Asymmetrisk embedding** via instruktions-prefix (`query:`/`passage:`, som i fx E5/BGE/GTE-modeller) for bedre krav→svar-matching. Dette er reelt "rod-fixet" til den asymmetri, hele Berigelse- og retrieval-fasen i dag kompenserer for med tags og forudberegnede matches: et jobkrav og et ansøgnings-svar er ikke symmetriske tekster (spørgsmål vs. svar), men embeddes i dag med samme model uden retningsspecifik instruktion. En asymmetrisk model, der embedder krav som "query" og svar som "passage", kunne potentielt reducere behovet for dele af tag-laget — men er ikke afprøvet endnu.
- Justering af k-værdier og similarity-thresholds (foreløbig sat ud fra stikprøver, ikke systematisk evalueret).
- Eksport af den færdige ansøgningstekst til PDF ( via Puppeteer).
