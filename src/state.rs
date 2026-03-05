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

    pub fn update_tension(&mut self, delta: i8) {
        let new_level = self.state.tension_level as i16 + delta as i16;
        self.state.tension_level = new_level.clamp(0, 100) as u8;
    }

    /// Переходит к следующему ходу, применяя затухание напряжённости.
    /// `decay` > 0 снижает глобальное напряжение на указанную величину.
    pub fn next_turn(&mut self, decay: i8) {
        self.state.turn_count += 1;
        if decay > 0 {
            self.update_tension(-decay);
        }
    }

    pub fn get_tension_level(&self) -> u8 {
        self.state.tension_level
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
    /// Значения: отрицательные = союзники, положительные = враги (диапазон -100..=100).
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
