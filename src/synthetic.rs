//! Generador de conversaciones sintéticas en español para entrenar el modelo.
//!
//! Cada muestra usa exactamente el mismo formato que `gpt_chat` en modo chat:
//! `<|user|> pregunta <|eot|>` seguido de `<|assistant|> respuesta <|eot|>`.

use crate::rng::Rng;

const CAPITALS: &[(&str, &str)] = &[
    ("España", "Madrid"),
    ("Francia", "París"),
    ("Italia", "Roma"),
    ("Alemania", "Berlín"),
    ("Portugal", "Lisboa"),
    ("Reino Unido", "Londres"),
    ("Japón", "Tokio"),
    ("China", "Pekín"),
    ("Argentina", "Buenos Aires"),
    ("México", "Ciudad de México"),
    ("Brasil", "Brasilia"),
    ("Chile", "Santiago"),
    ("Perú", "Lima"),
    ("Colombia", "Bogotá"),
    ("Egipto", "El Cairo"),
    ("Marruecos", "Rabat"),
    ("Canadá", "Ottawa"),
    ("Australia", "Canberra"),
    ("India", "Nueva Delhi"),
    ("Grecia", "Atenas"),
];

const ANIMALS: &[(&str, &str, &str)] = &[
    ("el", "perro", "guau"),
    ("el", "gato", "miau"),
    ("la", "vaca", "muu"),
    ("la", "oveja", "bee"),
    ("el", "pato", "cuac"),
    ("la", "gallina", "cocorocó"),
    ("el", "caballo", "hiiii"),
    ("la", "rana", "croac"),
    ("el", "lobo", "aullido"),
    ("el", "pájaro", "pío"),
];

const COLORS: &[(&str, &str)] = &[
    ("el cielo", "azul"),
    ("la hierba", "verde"),
    ("la nieve", "blanco"),
    ("el carbón", "negro"),
    ("la sangre", "rojo"),
    ("el sol", "amarillo"),
    ("una naranja", "naranja"),
    ("el chocolate", "marrón"),
    ("una violeta", "violeta"),
];

const TRANSLATIONS: &[(&str, &str)] = &[
    ("hola", "hello"),
    ("adiós", "goodbye"),
    ("gracias", "thank you"),
    ("sí", "yes"),
    ("no", "no"),
    ("agua", "water"),
    ("casa", "house"),
    ("libro", "book"),
    ("amigo", "friend"),
    ("gato", "cat"),
    ("perro", "dog"),
    ("sol", "sun"),
    ("luna", "moon"),
    ("rojo", "red"),
    ("azul", "blue"),
];

const WORDS: &[&str] = &[
    "gato", "perro", "casa", "libro", "mesa", "silla", "ventana", "árbol", "ciudad", "música",
    "escuela", "familia", "amigo", "camino", "jardín", "nube", "playa", "montaña", "estrella",
    "coche",
];

fn pick<'a, T>(rng: &mut Rng, values: &'a [T]) -> &'a T {
    &values[rng.below(values.len())]
}

fn conversation(question: &str, answer: &str) -> String {
    format!("<|user|> {question} <|eot|>\n<|assistant|> {answer} <|eot|>\n")
}

fn greeting(rng: &mut Rng) -> String {
    let question = pick(
        rng,
        &[
            "hola",
            "buenos días",
            "buenas tardes",
            "¿qué tal?",
            "¿cómo estás?",
            "hola, ¿cómo va todo?",
        ],
    );
    let answer = pick(
        rng,
        &[
            "Hola. ¿En qué puedo ayudarte?",
            "Buenos días. ¿Qué necesitas?",
            "Estoy bien, gracias. ¿Y tú?",
            "Todo bien. ¿En qué te ayudo?",
            "Hola, ¿qué quieres saber?",
        ],
    );
    conversation(question, answer)
}

fn capital(rng: &mut Rng) -> String {
    let (country, city) = pick(rng, CAPITALS);
    let question = match rng.below(3) {
        0 => format!("¿Cuál es la capital de {country}?"),
        1 => format!("Dime la capital de {country}."),
        _ => format!("¿Sabes cuál es la capital de {country}?"),
    };
    let answer = match rng.below(2) {
        0 => format!("La capital de {country} es {city}."),
        _ => format!("{city} es la capital de {country}."),
    };
    conversation(&question, &answer)
}

fn arithmetic(rng: &mut Rng) -> String {
    match rng.below(4) {
        0 => {
            let left = 1 + rng.below(20);
            let right = 1 + rng.below(20);
            conversation(
                &format!("¿Cuánto es {left} + {right}?"),
                &format!("{left} + {right} = {}.", left + right),
            )
        }
        1 => {
            let left = 1 + rng.below(20);
            let right = 1 + rng.below(left);
            conversation(
                &format!("¿Cuánto es {left} - {right}?"),
                &format!("{left} - {right} = {}.", left - right),
            )
        }
        2 => {
            let left = 1 + rng.below(12);
            let right = 1 + rng.below(12);
            conversation(
                &format!("¿Cuánto es {left} por {right}?"),
                &format!("{left} por {right} = {}.", left * right),
            )
        }
        _ => {
            let left = 1 + rng.below(20);
            let right = 1 + rng.below(20);
            conversation(
                &format!("Suma {left} y {right}."),
                &format!("El resultado de la suma es {}.", left + right),
            )
        }
    }
}

fn capitalized(text: &str) -> String {
    let mut characters = text.chars();
    match characters.next() {
        Some(first) => first.to_uppercase().collect::<String>() + characters.as_str(),
        None => String::new(),
    }
}

fn animal(rng: &mut Rng) -> String {
    let (article, animal, sound) = pick(rng, ANIMALS);
    let question = match rng.below(3) {
        0 => format!("¿Qué sonido hace {article} {animal}?"),
        1 => format!("¿Cómo hace {article} {animal}?"),
        _ => format!("¿Qué dice {article} {animal}?"),
    };
    let subject = format!("{} {animal}", capitalized(article));
    let answer = match rng.below(2) {
        0 => format!("{subject} hace «{sound}»."),
        _ => format!("{subject} dice «{sound}»."),
    };
    conversation(&question, &answer)
}

fn color(rng: &mut Rng) -> String {
    let (thing, color) = pick(rng, COLORS);
    conversation(
        &format!("¿De qué color es {thing}?"),
        &format!("{thing} es de color {color}."),
    )
}

fn counting(rng: &mut Rng) -> String {
    let limit = 3 + rng.below(7);
    let numbers: Vec<String> = (1..=limit).map(|value| value.to_string()).collect();
    conversation(
        &format!("Cuenta del 1 al {limit}."),
        &format!("{}.", numbers.join(", ")),
    )
}

fn parity(rng: &mut Rng) -> String {
    let number = 1 + rng.below(30);
    let parity = if number % 2 == 0 { "par" } else { "impar" };
    conversation(
        &format!("¿El número {number} es par o impar?"),
        &format!("El número {number} es {parity}."),
    )
}

fn translation(rng: &mut Rng) -> String {
    let (spanish, english) = pick(rng, TRANSLATIONS);
    conversation(
        &format!("¿Cómo se dice «{spanish}» en inglés?"),
        &format!("«{spanish}» en inglés es «{english}»."),
    )
}

fn initial_letter(rng: &mut Rng) -> String {
    let word = pick(rng, WORDS);
    let initial = word.chars().next().unwrap_or('?');
    conversation(
        &format!("¿Con qué letra empieza «{word}»?"),
        &format!("«{word}» empieza con la letra {initial}."),
    )
}

fn time_facts(rng: &mut Rng) -> String {
    let (question, answer) = pick(
        rng,
        &[
            (
                "¿Cuántos días tiene una semana?",
                "Una semana tiene 7 días.",
            ),
            ("¿Cuántas horas tiene un día?", "Un día tiene 24 horas."),
            (
                "¿Cuántos minutos tiene una hora?",
                "Una hora tiene 60 minutos.",
            ),
            ("¿Cuántos meses tiene un año?", "Un año tiene 12 meses."),
            ("¿Cuántos días tiene un año?", "Un año tiene 365 días."),
            (
                "¿Cuántos segundos tiene un minuto?",
                "Un minuto tiene 60 segundos.",
            ),
        ],
    );
    conversation(question, answer)
}

fn politeness(rng: &mut Rng) -> String {
    let (question, answer) = pick(
        rng,
        &[
            ("gracias", "De nada."),
            ("muchas gracias", "No hay de qué."),
            ("adiós", "Hasta luego."),
            ("hasta pronto", "Hasta pronto."),
            ("buenas noches", "Buenas noches, descansa."),
        ],
    );
    conversation(question, answer)
}

fn identity(rng: &mut Rng) -> String {
    let question = pick(
        rng,
        &[
            "¿Quién eres?",
            "¿Qué eres?",
            "¿Cómo te llamas?",
            "¿Qué puedes hacer?",
        ],
    );
    let answer = pick(
        rng,
        &[
            "Soy un modelo pequeño entrenado con datos sintéticos.",
            "Soy un mini GPT y respondo preguntas sencillas.",
            "Me llamo mini GPT y estoy aprendiendo español.",
            "Puedo saludar, hacer cuentas sencillas y responder datos básicos.",
        ],
    );
    conversation(question, answer)
}

fn unknown(rng: &mut Rng) -> String {
    let question = pick(
        rng,
        &[
            "¿Cuál es la capital de Marte?",
            "¿Cuántos habitantes tiene la Luna?",
            "¿Qué hay dentro de un agujero negro?",
            "¿Cuál es el animal más rápido del universo?",
            "¿Qué desayuna un robot?",
        ],
    );
    let answer = pick(
        rng,
        &[
            "No lo sé.",
            "No tengo esa información.",
            "No puedo responder eso con seguridad.",
        ],
    );
    conversation(question, answer)
}

fn sample(rng: &mut Rng) -> (&'static str, String) {
    match rng.below(13) {
        0 => ("saludos", greeting(rng)),
        1 => ("capitales", capital(rng)),
        2 => ("aritmética", arithmetic(rng)),
        3 => ("animales", animal(rng)),
        4 => ("colores", color(rng)),
        5 => ("conteo", counting(rng)),
        6 => ("paridad", parity(rng)),
        7 => ("traducción", translation(rng)),
        8 => ("letras", initial_letter(rng)),
        9 => ("tiempo", time_facts(rng)),
        10 => ("cortesía", politeness(rng)),
        11 => ("identidad", identity(rng)),
        _ => ("desconocido", unknown(rng)),
    }
}

/// Genera `count` conversaciones etiquetadas con su categoría.
pub fn generate_tagged(count: usize, seed: u32) -> Vec<(&'static str, String)> {
    let mut rng = Rng::new(seed);
    (0..count).map(|_| sample(&mut rng)).collect()
}

/// Genera `count` conversaciones listas para escribir en el corpus.
pub fn generate(count: usize, seed: u32) -> Vec<String> {
    generate_tagged(count, seed)
        .into_iter()
        .map(|(_, conversation)| conversation)
        .collect()
}
