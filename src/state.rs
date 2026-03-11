use crate::types::{CountryId, WorldState};

pub struct StateManager {
    pub state: WorldState,
}

impl StateManager {
    pub fn new() -> Self {
        Self {
            state: WorldState::default(),
        }
    }

    /// Инициализирует счётчики страны: очки влияния = 0, стабильность = 70.
    pub fn init_country(&mut self, id: &CountryId) {
        self.state.influence_scores.entry(id.0.clone()).or_insert(0);
        self.state.stability.entry(id.0.clone()).or_insert(70);
    }

    pub fn update_tension(&mut self, delta: i8) {
        let new_level = self.state.tension_level as i16 + delta as i16;
        self.state.tension_level = new_level.clamp(0, 100) as u8;
    }

    /// Переходит к следующему ходу, применяя затухание напряжённости.
    pub fn next_turn(&mut self, decay: i8) {
        self.state.turn_count += 1;
        if decay > 0 {
            self.update_tension(-decay);
        }
    }

    pub fn get_tension_level(&self) -> u8 {
        self.state.tension_level
    }

    /// Изменяет очки влияния актора и (если указана) цели.
    pub fn apply_influence(&mut self, actor: &CountryId, actor_delta: i32, target: Option<&CountryId>, target_delta: i32) {
        let actor_score = self.state.influence_scores.entry(actor.0.clone()).or_insert(0);
        *actor_score += actor_delta;

        if let Some(t) = target {
            let target_score = self.state.influence_scores.entry(t.0.clone()).or_insert(0);
            *target_score += target_delta;
        }
    }

    /// Изменяет стабильность одной страны (clamp 0..=100).
    pub fn update_stability(&mut self, id: &CountryId, delta: i8) {
        let entry = self.state.stability.entry(id.0.clone()).or_insert(70);
        *entry = (*entry as i16 + delta as i16).clamp(0, 100) as u8;
    }

    /// Пересчитывает стабильность в конце хода:
    /// - Страны, чьи очки ниже среднего на > 10: -5 стабильности (внутреннее давление)
    /// - Все страны: +2 (естественное восстановление)
    pub fn tick_stability(&mut self, country_ids: &[CountryId]) {
        let scores: Vec<i32> = country_ids
            .iter()
            .map(|id| *self.state.influence_scores.get(&id.0).unwrap_or(&0))
            .collect();

        let avg = if scores.is_empty() { 0 } else { scores.iter().sum::<i32>() / scores.len() as i32 };

        for id in country_ids {
            let score = *self.state.influence_scores.get(&id.0).unwrap_or(&0);
            let entry = self.state.stability.entry(id.0.clone()).or_insert(70);
            // Восстановление
            *entry = (*entry as i16 + 2).clamp(0, 100) as u8;
            // Давление от отставания
            if score < avg - 10 {
                *entry = (*entry as i16 - 5).clamp(0, 100) as u8;
            }
        }
    }

    /// Возвращает канонический ключ для пары стран (всегда в алфавитном порядке).
    fn relationship_key(a: &CountryId, b: &CountryId) -> String {
        if a.0 <= b.0 {
            format!("{}|{}", a.0, b.0)
        } else {
            format!("{}|{}", b.0, a.0)
        }
    }

    /// Обновляет уровень враждебности между двумя странами.
    pub fn update_relationship(&mut self, from: &CountryId, to: &CountryId, delta: i8) {
        let key = Self::relationship_key(from, to);
        let entry = self.state.relationships.entry(key).or_insert(0);
        *entry = (*entry as i16 + delta as i16).clamp(-100, 100) as i8;
    }

    /// Возвращает текущий уровень враждебности между двумя странами (0 по умолчанию).
    pub fn get_relationship(&self, a: &CountryId, b: &CountryId) -> i8 {
        let key = Self::relationship_key(a, b);
        *self.state.relationships.get(&key).unwrap_or(&0)
    }
}
