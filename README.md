# projektHireMe
# Intro

## motivation

### jobsøgning

**Projekt Hire Me — eller hvordan jeg lærte at avle mine jobansøgninger**
Der er ikke så meget sjovt ved at søge job som ledig. Man skal søge mindst 6 job om måneden, uanset om der overhovedet er så mange relevante opslag at finde — hverken på internettet eller i ens netværk. Som tiden går, bliver det mere og mere repetitivt at formulere, hvor motiveret man er for at arbejde med softwareudvikling inden for netop dén branche man søger ind i, og at beskrive hvordan ens egne kompetencer og kvaliteter matcher dem der efterspørges i opslaget.

Ofte sad jeg med det problem, at jeg allerede mange gange før havde beskrevet, at jeg kunne matche en bestemt efterspurgt kompetence — men i hvilken ansøgning var det nu, jeg skrev det? Det tog ofte længere tid at finde de gamle beskrivelser frem end bare at skrive dem igen. Så som softwareingeniør blev jeg nødt til at automatisere så meget som muligt af disse arbejdsgange i jobsøgningsprocessen.

Med LLM'ers fremgang, og hvor lette de er blevet at tilgå og anvende i dag — også for én der ikke er uddannet direkte i data engineering eller machine learning — er det oplagt at prompt-engineere sig ud af dette slidsomme, manuelle ansøgningsskriveri.

Jeg har altså ikke nøjedes med at gå ét metaniveau op og lave en fast copy-paste-prompt til at generere nye ansøgninger. I stedet har jeg, ved hjælp af agentic coding/vibecoding/chat, udviklet et RAG-setup, som dynamisk sammensætter en prompt skræddersyet til netop dét jobopslag, jeg vil ansøge.

### Forrige arbejdsmetoder
Før dette projekt bestod arbejdsgangen af: manuel samling af tekstbidder fra tidligere ansøgninger ved hjælp af en LLM, samt et lille script der brugte Jaccard-overlap (mængde-lighed mellem ordsæt) til at finde tekstbidder, med en prompt som slutresultat. Begge tilgange krævede stadig, at jeg selv huskede eller ledte efter, hvilke gamle ansøgninger der dækkede hvilke kompetencer — den del automatiserede de ikke.

## Use Case:
Jeg vil som bruger kunne copy-paste teksten fra et jobopslag som input, og som output få returneret en ansøgningstekst, som jeg selv ville have skrevet den.

## Problemstilling:
Hvordan kan tidligere formuleringer i ansøgninger, der matcher bestemte kompetencer og kvaliteter efterspurgt i et givent jobopslag, søges frem, kombineres og genbruges i en ny ansøgning?

### Kerneidé:
Tidligere jobopslag sammen med de ansøgninger, jeg har sendt til dem, betragtes som krav-svar-par. Et RAG-setup udleder kravene i et givent jobopslag, finder matchende krav i de tidligere jobopslag, og returnerer krav-svar-matches som guidende eksempler i den prompt, der gives til en LLM, som skal generere en ansøgning til det nye opslag.


### Tekniske krav:
- I første omgang et script der kan kaldes i terminalen; på sigt en webside-front.
- Offline LLM-inferens, så vidt muligt.
- En vis grad af kvalitet i outputteksten.

### Resourser
- 50+ jobopslag og ansøgninger i krav-svar-par, som RAG-data.
- Softwareudvikler-kompetencer og -erfaring; Python-scripting og SQL er særligt relevante i dette projekt.
- NVIDIA GeForce RTX 3070
- Internet 780 (Mbps)

# Løsning:

## Pipeline:

### Oversigt:

Løsningen, som den ser ud i skrivende stund, er beskrevet herunder, som den pipeline løsningen udgør. De enkelte faser er beskrevet nærmere i [Building](#building):

**1. Ingestion** ([afsnit](#ingestion))
- Gennemgår alle opslag og ansøgninger parvis, som en "case"; søger filer via bestemte navnemønstre.
- Udtrækker ren tekst pr. fil — dels fra `.txt`, mest fra HTML (som jeg har brugt til at generere PDF'er ud fra); trimmer og samler strings.
- Prompter en lokal LLM (Ollama) om at opdele hver tekst i rensede, semantiske hele sætninger 
- Gemmer sætningerne i rækkefølge i en JSON-fil (rækkefølgen er nødvendig senere, når sætningerne skal grupperes i paragraffer).
- Upserter hver case, så allerede behandlede cases ikke duplikeres ved genkørsel.
- Kanonisering: hver case og sætning kanoniseres med et stabilt id.
- Regelbaseret rensning: fjerner kendte boilerplate-/rekrutteringsfraser fra både rå tekst og sætningslister.
- Statistisk rensning: filtrerer sætninger der er unikt hyppige på tværs af hele tekstkorpusset (corpus-wide distinctiveness), via embedding + FAISS self-similarity search.

**2. Semantic Chunking** ([afsnit](#sematisk-chunking))
- Sammenlægger sætninger til semantisk sammenhængende paragraffer — via embedding + FAISS self-similarity search mellem naboende sætninger.
- Opslagenes og ansøgningernes hele tekster og paragraffer indekseres relationelt.

**3. Persistering i PostgreSQL (system-of-record)** ([afsnit](#persistering-i-postgresql))
- Opretter/synkroniserer tabeller
- Gemmer cases, dokumenter og paragraffer og relationer
- Gemmer de forudberegnede krav→svar-paragrafmatches.
- Gennemgang af tabellerne og deres relationer, samt ER-diagram.

**4. Berigelse** ([afsnit](#berigelse))
- Embedding matching: krav-svar-lighed inden for hver case — opslags- og ansøgningstekst er allerede matchet på case-niveau; anvender embedding similarity til også at beregne ligheden mellem opslags-paragraffer og ansøgnings-paragraffer.
- Tag matching: hardcodede taksonomiske tags med seed-tekst-eksempler til bedre embedding similarity search; beregner taksonomisk tag-lighed via embedding-similaritet; gemmer tags og similarity-scores i databasen.

**5. FAISS-indeksering** ([afsnit](#faiss-indeksering))
- Genererer embeddings for dokumenter og paragraffer i fire Postgres-collections.
- Embedder og L2-normaliserer teksterne pr. collection.
- Bygger ét `IndexFlatIP`-indeks pr. collection og skriver det til disk.
- Gemmer `faiss_id_map`, så en FAISS-position kan slås op til den oprindelige databaserække.

**6. Query-time retrieval** ([afsnit](#retrieval))
- Input: tekstfil med et nyt jobopslag.
- Sætningsopdeler og genbruger paragraf-segmenteringen fra trin 2 til at danne query-paragraffer.
- Kompetence-matching: søger opslaget for ord/varianter fra en fast kompetenceliste.
- Tagger query-teksten mod de samme tag-prototyper som i Berigelse.
- Dokument-niveau retrieval: FAISS-søgning i opslags-tabellen, henter tilhørende hele ansøgning, reranker efter tag-overlap og vector-score.
- Paragraf-niveau retrieval: FAISS-søgning i opslags-paragraf-tabellen pr. query-paragraf.
- Diversitets-udvælgelse: vælger forskellige jobparagraffer og deres bedst matchende svarparagraf, så prompten ikke domineres af ét tema.
- Henter separat korte ansøgnings-paragraffer, der direkte indeholder de matchede kompetenceord.

**7. Promptgenerering** ([afsnit](#promptgenerering))
- Sammensætter den endelige few-shot prompt: nyt jobopslag, tilladt kompetenceliste, korte kompetence-eksempler, hele tidligere eksempler (opslag + ansøgning), krav→svar-paragrafpar.
- Tilføjer guardrails: sprog, tone, stil, kun dokumenterede teknologier, forbud mod opdigtede kompetencer, konkrete formuleringsregler.
- Skriver resultatet til en fil, klar til LLM-inferens.

### Building

#### Ingestion

Som udgangspunkt er alle opslags- og ansøgningdokument-par allerede organiseret i egne mapper — udfordringen er at normalisere og formatere teksten i dokumenterne, så de opfylder de krav, resten af pipelinen stiller: entydige, rene, hele sætninger, gemt i en stabil rækkefølge.
(rækkefølgen bruges direkte i næste fase til at afgøre, hvilke sætninger der hører sammen i en paragraf).

**Teori.** Fri tekst fra jobopslag og ansøgninger er fyldt med støj: overskrifter uden punktum, opremsninger, forkortelser, kontaktinfo midt i teksten osv. 
En simpel regex er ikke tilstrækkelig til at opdele en tekst i sætninger, f.eks. vil `.`/`!`/`?` knække let på forkortelser eller vil overspringe bullet-punkter uden slutpunktum. 

Derfor bruges her i stedet en lokal LLM til selve sætningsopdelingen — men *kun* til opdelingen, ikke til at omskrive eller opsummere teksten. 
Ollama kaldes til at prompte LLM modellen `hf.co/unsloth/gemma-4-E4B-it-qat-GGUF:UD-Q4_K_XL` med følgende prompt, hvor begrænsingerne er sat helt konkret og begrebet "sætning" er defineret:

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

Med følgende Options:

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

 Hvor `temperature: 0.0`, skruer helt ned for kreativiteten, så input konsekvent giver samme output — determinisme er en forudsætning for at kunne bygge en pipeline, man kan stole på og genkøre.
 Og `"system": "Du er en dygtig tekstsegmenteringsassistent. Din opgave ...` er System prompten

**Ræsonnement — hvorfor en LLM og ikke bare rensning i kode.** Alternativet, en hardcoded regex/NLP-splitter, ville kræve at forudse enhver særskrivning i danske jobopslag (forkortelser, tal, bindestregs-sammensætninger), og rent praktisk kræve løbende udviddelse af splitting-regler. En LLM med en præcis instruktion generaliserer bedre til den slags uforudsete formatering — så længe instruktionen selv er præcis. 

**Praktisk erfaring — "helsætning"-fælden.** Et konkret eksempel fra arbejdsforløbet: da prompten til Ollama i en tidlige iteration bad om at splitte teksten i "sætninger" uden at definere ordet "sætning", var modellens fortolkning af hvad der udgør én sætning (fx om en overskrift eller et enkelt bullet-punkt talte som "en sætning") uforudsigelig nok til at give enten for aggressiv eller for konservativ opdeling, samt lejlighedsvis ugyldig returneret JSON. 
Løsningen var at gøre kravende så eksplicitte som muligt i selve prompten.

**Regelbaseret og statistisk rensning + kanonisering (`clean_and_canonize.py`).**

Selv efter LLM-sætningsopdelingen er teksten stadig fyldt med boilerplate — hilsner, kontaktoplysninger, rekrutterings-fraser — samt sætninger der er så generiske, at de går igen på tværs af helt urelaterede jobopslag. Denne rensning, sammen med tildelingen af stabile id'er til hver case, arbejder på de rå, endnu ikke grupperede sætninger og hører derfor konceptuelt til Ingestion, selvom den historisk lå i det senere chunking-script.

`remove_phrases`-listen: en regex-liste over kendte danske rekrutterings-/boilerplate-fraser ("send os en ansøgning", telefonnumre, URL'er, "med venlig hilsen", "cpr-nummer", "frokostordning"/"massageordning"/"pensionsordning", ligestillings-boilerplate m.fl.) — en billig, regelbaseret første rensning, før den dyrere statistiske rensning nedenfor.

`cleanup(cases_json, embeddingmodel)`:
- fjerner sætninger der matcher `remove_phrases` fra både sætningslister og rå tekst (erstattes med `" [] "`);
- tildeler hver case et `case_id` (`"c" + index`) — kanonisering;
- **corpus-wide distinctiveness-filtrering**: samler alle `job_sentences` på tværs af *alle* cases i én flad liste med et `flatID_to_caseID_mapping`, embedder dem alle, bygger ét `faiss.IndexFlatIP` over hele listen, og selv-søger hver sætning mod sine `k=50` nærmeste naboer;
- for hver sætning tælles, hvor mange forskellige cases der har en næsten-identisk match (similarity ≥ 0.85), og `distinctiveness = 1.0 - (antal matchende cases / samlet antal cases)`;
- er en sætnings distinctiveness under 0.9, fjernes den — den er med andre ord for generisk/hyppig på tværs af hele korpusset til at være informativ (en statistisk, embedding-baseret udgave af samme idé som regex-listen ovenfor, blot uden en hardcodet fraseliste).

`main()` læser `extracted_sentences.json` (`EXTRACTED_SENTENCES_JSON`), kører `cleanup()`, og skriver resultatet til `cleaned_sentences.json` (`CLEANED_SENTENCES_JSON`) — samme case-/sætningsstruktur som `paragraphize()` (næste fase) forventer som input.

#### Sematisk Chunking
**Teori — vektor, embedding, cosine similarity.** 
En grundlæggende problemstilling inden for sprogteknologi (Natural Language Processing (NLP)) og LLM teknologien er hvordan man repræsenterer en tekst numerisk, så *den meningsfulde sammenhæng* mellem tekstens enkelte dele bevares, og en computer kan regne på den.
Det er nemt nok at give alle bogstaver alfabetet en unik numerisk værdi, at *kvantificere* dem; en simpel måde er f.eks. at give dem alle sin egen unikke talrepræsentation, hvor a = 1, b = 2 osv. her vil alle unikke ord få en unik talkombination. Men det hjælper ikke på at kunne "regne ud" hvilken kombination af ord, der sammen giver mening i en tekst.
Bogstaver er, udover enkeltbogstavsord som "i" "å" og "ø", ikke meningsbærende i sig selv. Den mindste enhed af mening i en tekst er for mennesker *ord*,  for LLM'er er det *tokens*, som også inkluderer hele ord fra ordbogen, men også ord-bidder, dele af ord, som går igen i et sprog f.eks. gramatiske endelser og "for-", "til-" og "af-" på dansk. Det er lettest at forstå *tokens*=*ord*, selv om det er en forsimpling af tingene. 

Kvantificeringen af disse tokens, eller ord om man vil, går ud på at tillægge allesammen en unik vektor. 
Hvis læseren ikke kender begrebet *vektor*, så kan det være tilstrækkeligt at forstå en vektor som en samling af tal i en bestemt rækkefølge, fx `[0.12, -0.45, 0.88, …]` eller `[-0.67, 57.11, 40.01, …]`.
Dette er et alsidigt matematisk redskab, som i denne her sammenhæng bruges til at tillægge en token grader af forskellige *egenskaber*, hvert tal i vektoren angiver simpelthen hvor meget denne token bærer en vis egenskab. 
Disse egenskaber er resultatet af at embbeddingmodellen er blevet trænet på helt vilde mængder tekst, og har fundet, at der gennemgående er noget der statistisk set gentager sig ved brugen af disse ord i tekster, mao. hvordan ordene relaterer sig til hinanden i en tekst, ikke ordene i sig selv. Et godt intuitivt eksempel er de egenskaber der går igen i ordene "mand" "husbond" og "konge" og dem der går igen i "kvinde" "kone" og "dronning". Nogle egenskaber går igen i højere grad blandt nogle ord, men i lavere grad i andre ord. Disse egenskaber er repræsenteret af en eller flere tal-positioner i ordenes vektor, og jo mere meningsmæssigt en token er lig en anden token jo tættere er tallenes indbyrdes forhold. Men egenskaberen er ikke nødvendigvis særligt forståelige for mennesker, som i eksemplet ovenfor, de skal bare være statistisk signifikante for at kunne bevare mening.

Dette er et forsøg på at forklare overordnet den grundlæggende kvantificeringen af tekst, kaldt embedding, hvor altså teksten deles op og lægges i vektorer. En hel tekst, altså en rækkefølge af tokens, kan også repræsenteres af en enkelt vektor, som er udregnet fra hver af dens tokens vektorer. Så en teksts egen vektor er altså udrenget fra alle dens tokens vektorer. 
For en bestemt embeddingmodel har vektorene, som den anvender, et bestemt antal tal, en bestemt længde.  
I dette har projekt har jeg anvendt en embeddingmodel der hedder `paraphrase-multilingual-MiniLM-L12-v2`, som anvender vektorer, der er 384 tal lange, den har 384 dimensioner, som det kaldes. Det er en lille en af slagsen, som jeg har valgt netop fordi jeg prøver at køre så meget af systemet lokalt på min egen computer. Der findes også embeddingmodeller med vektorer der har 3072 dimensioner. 

Embeddingmodeller bruges til at kvantificere tekst, et trin der også kaldes encoding, så en computer kan sammenligne dens meningsfulde lighed med andre teksters; de udgør dog ikke i sig selv hele generative LLM'er, men encoding indgår også som et trin inde i en generativ LLM. Generative LLM'er som Claude og ChatGPT har et internt trin der omsætter tokens til vektorer, men det er trænet sammen med resten af modellen som én helhed, i modsætning til den her anvendte, separat trænede embeddingmodel. I den anden ende af sådanne en generative LLM-modeler bliver tal løbende omsat til tokens og dermed tekst igen, det trin kaldes så decoding. Givet en kvantificeret tekst, skal den generativ LLM-model regne ud hvad den mest sandsynlige næste token bør være for at tekstens mening bevares.

En embeddingmodel kan altså ikke generere tekst, den kan kun analysere deres statistiske lighed, deres semantiske mening. Men fordi den
netop er kvantificeret i vektorer, det er bare en masse tal, så kan man regne ud hvor meget de meningsmæssigt minder om hinanden. 
Den mest gængse udregning af ligheden kaldes cosinus-lighed, på engelsk *cosine similarity*, som kommer af, at inden for geometri, der kan vektorer beskrives som at være en længde med en retning i et koordinatsystem. Hvad har "længder", "retninger" og "koordinatsystemer" med semantisk mening at gøre? Jo, hvis vi forestiller os, at man skulle plotte ord ind i på et kort over mening, så er det oplagt at jo mere ord ligner hinanden, jo tættere vil de være placeret på et kort, som man jo aflæser med et koordinatsystem, og det er lige netop det, der er tilfældet med embedding af tekst, lighed er nærhed i koordinatsystemet. En vektors retning og længde er afgjort af tallene, som den består af, og de tal angiver, som beskrevet ovenfor egenskaber ved ordet eller sætningen, som tilsammen udgør den semantiske mening.
Vi er dog kun interesseret i retningen af vektorerne, for længden af en vektor, nemlig hvor store tallene er i den, siger mere om hvor lang teksten er end den egentlige mening som den bære på. Retningen afgøres af de indbyrdes forhold mellem alle tallene i vektoren.
For at "omforme" to teksters, eller ord (tokens), vektor så længden er ligegyldig, så *normaliserer* dem, de får samme længde, og kun retningen er til forskel. Hvis man så betragter de to vektorer som to lige lange linier, der starter i 0 i koordinatsystemet, så vil vinklen
mellem dem, hvor meget de peger i samme retning, indikerer hvor meget tallene, altså egenskaberne, altså den semantiske mening ligner hinanden.


**Teori — hvorfor semantisk chunking, og hvorfor ikke arbitrær chunking.** Chunking betyder at dele lang tekst op i mindre stykker, så retrieval bliver mere præcist og LLM'ens kontekst mindre støjet af irrelevant indhold. Den mest almindelige RAG-tilgang er arbitrær chunking: fast antal tegn eller tokens pr. chunk, uafhængigt af indhold. Det er ikke oplagt her, fordi jobopslag og ansøgninger i dette projekt er korte dokumenter (typisk nogle hundrede ord) — en fast tegn-vindue-chunking ville ofte skære midt i en sætning eller blande to urelaterede emner sammen i én chunk. Løsningen er i stedet at arbejde med sætningen som den mindste enhed (allerede sikret i Ingestion) og derefter gruppere sætninger til paragraffer ud fra deres *semantiske sammenhæng* snarere end en fast længde.

**v1 — naiv chunking + heldokument-embedding-match.** Det første forsøg embeddede hele dokumenter/paragraffer og matchede dem direkte. Det virkede — systemet fandt faktisk relevante uddrag fra databasen — men blandt de bedste matches optrådte også eksempler, der tydeligvis ikke var reelt relevante. Det afslørede en central indsigt: embedding som metode til at sammenligne tekst skal ikke gøres til mere, end det egentlig er — en kvantificering af *hele* teksten, intet mere, intet mindre. Den semantiske mening trækkes ud af modellen, men maskinen deler ikke nødvendigvis udviklerens eller brugerens opfattelse af, hvad der er semantisk vigtigt. Derfor fandt systemet tekster der "lød" ens, ikke kun tekster med sammenlignelige arbejdsopgaver — rekrutterings-lingoen ("vi tilbyder", "et godt arbejdsmiljø") lyder også ens på tværs af helt urelaterede jobopslag, og den tæller med i similarity-scoren, medmindre man enten fjerner fokus fra den (rensning) eller flytter fokus over på mere fagligt relevant mening (tags, se Berigelse).



#### Berigelse

Efter v1's diagnose (embedding alene finder tekster der lyder ens, ikke nødvendigvis tekster der reelt matcher fagligt) er systemet nødt til at understøtte RAG-forespørgslen med mere end rå embedding-similarity — dels en persisteret model for "hvilket svar passer bedst til hvilket krav", dels en model for "handler disse to tekster overhovedet om det samme". To uafhængige, komplementære former for match beregnes derfor her.

**Ræsonnement — hvorfor tag-matching blev nødvendig.** Et konkret eksempel under udviklingen viste problemet tydeligt: en sætning om "PET" (politiets efterretningstjeneste) blev embedding-matchet højt op mod en sætning om "MinRefusion" (et fagsystem) — to helt urelaterede domæner, der blot delte overfladisk sprogbrug/genre. En ren embedding-similarity-score kan ikke i sig selv skelne mellem "de her to tekster ligner hinanden i stil" og "de her to tekster handler om det samme emne". Løsningen er en lukket, håndskrevet taksonomi af tags — et fast, kontrolleret vokabular, som ikke kan snydes af overfladisk sproglig lighed på samme måde som fri embedding-similarity kan.

**Taksonomisk tagging.**

Spørgsmålet et tag besvarer er: *"Handler disse to tekster overhovedet om samme emne/domæne?"* (Python vs. cybersikkerhed vs. salg) — et diskret, kategorisk svar, robust over for stilforskelle. Et tag som "programmeringssprog" er sandt, uanset om teksten er formuleret som et krav i et opslag eller som en erfaring i en ansøgning — hvor embedding-similarity ville kunne variere en del bare på formuleringens stil.



**embedding similarity matching.**

Spørgsmålet embedding-similarity besvarer, når to tekster allerede er inden for samme case, er anderledes end tag-matchets: *"Hvor specifikt/stærkt besvarer denne ene sætning netop dette ene krav?"* — en kontinuerlig score, der fanger nuancer inden for samme tag-kategori. To tekster kan begge være tagget "Python-erfaring", men "5 års Python-erfaring i finansielle systemer" matcher alligevel bedre til et FinTech-Python-krav end "lidt Python i et skoleprojekt" — den forskel fanges kun af den kontinuerlige similarity-score, ikke af tagget alene.


**Ræsonnement — hvorfor kun inden for samme case.** At begrænse denne beregning til én case ad gangen (ikke hele korpusset) er bevidst: antagelsen er, at inden for én allerede matchet case er en jobparagraf og den/de sætninger i den tilhørende ansøgning, der besvarer netop det krav, den mest pålidelige grundsandhed, systemet har til rådighed — fordi et menneske (mig selv) faktisk brugte den ansøgning til at besvare netop det opslag.



#### Persistering i PostgreSQL

PostgreSQL er systemets "system of record": de faktiske tekster, id'er og relationer ligger her, og kan ikke uden videre genskabes, hvis de mistes. FAISS-indeksene (næste fase) er derimod bevidst behandlet som en flygtig, genopbyggelig afledning af data i Postgres — de kan til enhver tid slettes og genopbygges fra databasen, men ikke omvendt. Alle scripts, der skriver til Postgres, er skrevet idempotent (`ON CONFLICT ... DO UPDATE`), så en genkørsel aldrig skaber dubletter.

Nogle felter i skemaet — fx `tags_json`/`style_signals_json` på `job_paragraphs`/`app_paragraphs` i det officielle skema — står reelt tomme, fordi de faktiske tags i stedet lever i den mere fleksible `text_tags`-tabel (en normaliseret, polymorf kobling frem for en indlejret JSON-kolonne på selve raden). Det er et lille stykke skema-gæld: kolonnen blev designet ind tidligt "i tilfælde af", men den funktionalitet endte med at få sin egen tabel i stedet.

**Tabeloversigt.**

- `cases` — én række pr. case (jobopslag + ansøgning som ét par).
- `job_documents` / `app_documents` — hele rå-teksten pr. case, 1:1 med `cases`.
- `job_paragraphs` / `app_paragraphs` — de semantisk segmenterede paragraffer, 1:N pr. case.
- `job_app_paragraph_matches` — krav→svar-koblingen: `(job_paragraph_id, app_paragraph_id, similarity_score, rank)`, top-N pr. jobparagraf.
- `taxonomy_metadata`, `tags`, `tag_seeds` — selve taksonomien, synkroniseret fra `tags_taxonomy.json`.
- `text_tags` — den polymorfe kobling mellem en hvilken som helst tagget tekstenhed (`target_type`/`target_id`) og et tag, med score og `rank_in_category`.
- `faiss_id_map` — bro-tabellen mellem en FAISS-indeksposition og den oprindelige Postgres-række (se næste fase).


#### FAISS-indeksering

**Teori — fra tekst til søgbar vektor.** Kæden fra rå tekst til noget, man kan sammenligne tal-for-tal, går i store sprogmodeller typisk via: tekst → tokenizer → tokens (mindste tekst-enheder modellen regner med, ofte dele af ord) → token-id'er → en embedding-matrix der slår hvert token-id op som en vektor → transformer-lag der bearbejder tokenvektorerne i kontekst af hinanden. En *sentence embedding*-model (her en flersproget SentenceTransformer) gør noget lidt andet med det sidste trin: i stedet for at returnere én vektor pr. token, samler (poolers) den token-vektorerne til **én enkelt, fast-størrelse vektor for hele input-strengen** — det er den vektor, der gemmes og søges i, ikke token-vektorerne.

**Teori — hvad er FAISS.** FAISS (Facebook AI Similarity Search) er et bibliotek bygget specifikt til hurtig nearest-neighbor-søgning blandt store mængder vektorer. At gøre det "i hånden" med numpy (sammenligne en query-vektor mod hver gemt vektor én for én, eller ét stort matrix-produkt) er faktisk fuldt ud brugbart ved dette projekts skala (nogle tusinde vektorer) — men FAISS er standardværktøjet, der også skalerer til millioner af vektorer, og som giver et ensartet API til at bygge, gemme, indlæse og søge et indeks, uanset datamængde.

`IndexFlatIP` er den simpleste indekstype i FAISS: et eksakt (ikke-approksimeret), brute-force-indeks, der rangerer efter **inner product** (prikprodukt). Kombineret med at hver vektor L2-normaliseres før den tilføjes/søges (`faiss.normalize_L2`), bliver inner product matematisk identisk med cosine similarity — hvilket er hele grunden til, at normalisering sker konsekvent hver gang, en tekst embeddes nogen steder i denne kodebase.

**Ræsonnement — hvorfor fire separate indeks.** I stedet for ét stort, fælles indeks bygges der fire adskilte collections: `job_documents`, `app_documents`, `job_paragraphs`, `app_paragraphs`. Det er en direkte konsekvens af v1's erfaring (se Semantisk Chunking): blander man jobopslag og ansøgninger, eller hele dokumenter og paragraffer, i samme indeks, kan genre og længde dominere similarity-scoren frem for reel faglig relevans. Adskilte collections pr. (dokumenttype × granularitet) sikrer, at sammenligninger altid sker "samme genre mod samme genre".

### QUERY time

#### retrieval

Her bliver asymmetrien mellem jobopslag ("spørgsmål") og ansøgning ("svar") konkret: retrieval er ikke ét embedding-similarity-opslag, men flere adskilte opslag, der hver besvarer et andet spørgsmål, som derefter kombineres og rangeres — fordi intet enkelt "ligner dette"-signal er tilstrækkeligt alene.

To granulariteter bruges parallelt: **dokument-niveau** ("hvilken af mine tidligere ansøgninger læser bedst som en stilistisk skabelon til dette nye opslag") og **paragraf-niveau** ("hvilket tidligere krav ligner netop dette nye krav mest, og hvad svarede jeg dengang"). At `job_app_paragraph_matches` blev forudberegnet offline i Berigelse-fasen er præcis, hvad der gør dette trin til et billigt SQL-opslag i stedet for en dyr live-similarity-beregning mellem hver jobparagraf og hver app-paragraf i hele korpusset.

**Ræsonnement — diversitets-udvælgelse.** Uden yderligere filtrering kunne de bedste tag-overlap/vector-score-matches alle stamme fra samme jobparagraf eller samme tema — hvilket ville gøre de endelige few-shot-eksempler i prompten repetitive og skæve mod ét emne i stedet for at dække det nye opslags flere forskellige krav. Da antallet af paragraf-eksempler, prompten har råd til, er begrænset (jf. context-window-budgettet nedenfor), skal hver "plads" i prompten helst dække et *forskelligt* krav frem for nær-dubletter af samme krav. Løsningen: gruppér kandidatpar efter `job_paragraph_id`, behold kun distinkte jobparagraffer (rangeret efter `job_tag_overlap` → `vector_score`), og vælg for hver af dem uafhængigt den bedst matchende app-paragraf (efter `app_tag_overlap` → `mapping_score` → `vector_score`).

**Ræsonnement — kompetence-matching ved siden af tag/embedding.** Tag- og embedding-laget er bevidst fuzzy og probabilistisk — en styrke, fordi det generaliserer til formuleringer, systemet ikke har set før, men en svaghed, hvis man skal *garantere*, at et specifikt, konkret ord (fx "Kubernetes" eller "Scrum") bliver genkendt som til stede. En lille, håndholdt, deterministisk keyword-liste (`kompetencer_og_erfaringer_tags.txt`) lukker det hul: den giver LLM'en en eksplicit, forsvarlig "whitelist" af ord, den må bruge — samme guardrail-tankegang som i Promptgenerering, blot fra den modsatte vinkel: i stedet for kun at *forbyde* opdigtede ord, *tillader* denne liste aktivt de ord, der reelt findes i det nye opslag.

#### promptgenerering

**Teori — context window og token-budget.** Hver eneste bid, der er hentet frem ovenfor (kompetenceliste, korte kompetence-eksempler, hele dokumenter, paragrafpar), konkurrerer om det samme token-budget i prompten. En LLM's kontekstvindue er ikke uendeligt, og indhold placeret midt i en meget lang prompt har en dokumenteret tendens til at blive "glemt" eller vægtet lavere af modellen ("lost in the middle") sammenlignet med indhold i starten eller slutningen. Det er den direkte begrundelse for, at `document_k` som standard er sat lavt (1) og `paragraph_k` moderat (5) — ikke fordi flere eksempler ikke kunne være nyttige i teorien, men fordi hvert ekstra eksempel har en pris i kontekstplads og opmærksomhed.
I Ollama klienten kan man sætte "Context length" under Settings, hvor jeg nu har den til maks, fordi 
de prompt jeg får genereret pt. er meget tekst tunge og dermed indeholder mange tokens. Dette bør på sigt optimeres.

**Ræsonnement — guardrails mod hallucination.** "Hard requirements"-blokken i `build_prompt()` er den konkrete implementering af en gennemgående lære fra hele forløbet: en LLM skal eksplicit fortælles, hvad den *ikke* må gøre, lige så præcist som hvad den skal. 

Guardrailen har to sider, der begge er med: et *forbud* (opfind aldrig teknologier/kompetencer/erfaring, der ikke er dokumenteret i kildeteksterne) og en *tilladelse* (den eksplicitte kompetence-whitelist fra retrieval-fasen, som aktivt licenserer bestemte ord). Det er værd at være ærlig om, at den content/style/rejected-tredeling, der oprindeligt var tanken bag guardrails, i den nuværende implementering udmøntes som tekstlige instruktioner i prompten frem for som en bogstavelig tredelt eksempel-struktur.



#### Fremtidigt / endnu ikke implementeret

- Selve LLM-inferencen på den genererede prompt (lokal, offline) — endnu ikke koblet på; pipelinen stopper i dag ved `generated_prompt.txt`.
- **Asymmetrisk embedding** via instruktions-prefix (`query:`/`passage:`, som i fx E5/BGE/GTE-modeller) for bedre krav→svar-matching. Dette er reelt "rod-fixet" til den asymmetri, hele Berigelse- og retrieval-fasen i dag kompenserer for med tags og forudberegnede matches: et jobkrav og et ansøgnings-svar er ikke symmetriske tekster (spørgsmål vs. svar), men embeddes i dag med samme model uden retningsspecifik instruktion. En asymmetrisk model, der embedder krav som "query" og svar som "passage", kunne potentielt reducere behovet for dele af tag-laget — men er ikke afprøvet endnu.
- Justering af k-værdier og similarity-thresholds (foreløbig sat ud fra stikprøver, ikke systematisk evalueret).
- Eksport af den færdige ansøgningstekst til PDF (fx via Puppeteer).
