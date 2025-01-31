use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use crate::utils::{next_power_of_two, zeroed_f_vec};
use p3_air::{Air, BaseAir};
use p3_field::PrimeField32;
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::{ParallelBridge, ParallelIterator};
use sp1_core_executor::{ExecutionRecord, Program};
use sp1_derive::AlignedBorrow;
use sp1_stark::{
    air::{AirInteraction, InteractionScope, MachineAir, SP1AirBuilder},
    InteractionKind, Word,
};

pub const NUM_LOCAL_MEMORY_ENTRIES_PER_ROW: usize = 4;

pub(crate) const NUM_MEMORY_LOCAL_INIT_COLS: usize = size_of::<MemoryLocalCols<u8>>();

#[derive(AlignedBorrow, Debug, Clone, Copy)]
#[repr(C)]
struct SingleMemoryLocal<T> {
    /// The address of the memory access.
    pub addr: T,

    /// The initial shard of the memory access.
    pub initial_shard: T,

    /// The final shard of the memory access.
    pub final_shard: T,

    /// The initial clk of the memory access.
    pub initial_clk: T,

    /// The final clk of the memory access.
    pub final_clk: T,

    /// The initial value of the memory access.
    pub initial_value: Word<T>,

    /// The final value of the memory access.
    pub final_value: Word<T>,

    /// Whether the memory access is a real access.
    pub is_real: T,
}

#[derive(AlignedBorrow, Debug, Clone, Copy)]
#[repr(C)]
pub struct MemoryLocalCols<T> {
    memory_local_entries: [SingleMemoryLocal<T>; NUM_LOCAL_MEMORY_ENTRIES_PER_ROW],
}

pub struct MemoryLocalChip {}

impl MemoryLocalChip {
    /// Creates a new memory chip with a certain type.
    pub const fn new() -> Self {
        Self {}
    }
}

impl<F> BaseAir<F> for MemoryLocalChip {
    fn width(&self) -> usize {
        NUM_MEMORY_LOCAL_INIT_COLS
    }
}

impl<F: PrimeField32> MachineAir<F> for MemoryLocalChip {
    type Record = ExecutionRecord;

    type Program = Program;

    fn name(&self) -> String {
        "MemoryLocal".to_string()
    }

    // fn generate_dependencies(&self, _input: &ExecutionRecord, _output: &mut ExecutionRecord) {
    //     // Do nothing since this chip has no dependencies.
    // }

    // fn generate_trace(
    //     &self,
    //     input: &ExecutionRecord,
    //     _output: &mut ExecutionRecord,
    // ) -> RowMajorMatrix<F> {
    //     // Generate the trace rows for each event.
    //     let mut rows = input
    //         .memory_events
    //         .iter()
    //         .filter(|event| event.is_local())
    //         .map(|event| {
    //             let mut row = [F::zero(); NUM_LOCAL_MEMORY_COLS];
    //             let cols: &mut LocalMemoryCols<F> = row.as_mut_slice().borrow_mut();
    //             cols.shard = F::from_canonical_u32(input.public_values.execution_shard);
    //             cols.address = F::from_canonical_u32(event.address);
    //             cols.value = F::from_canonical_u32(event.value);
    //             cols.clock = F::from_canonical_u32(event.clock);
    //             cols.is_read = F::from_canonical_u32(event.is_read as u32);
    //             row
    //         })
    //         .collect::<Vec<_>>();

    //     // Pad the trace to a power of two depending on the proof shape in `input`.
    //     pad_rows_fixed(
    //         &mut rows,
    //         || [F::zero(); NUM_LOCAL_MEMORY_COLS],
    //         input.fixed_log2_rows::<F, _>(self),
    //     );

    //     RowMajorMatrix::new(rows.into_iter().flatten().collect::<Vec<_>>(), NUM_LOCAL_MEMORY_COLS)
    // }

    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            shard.get_local_mem_events().nth(0).is_some()
        }
    }

    fn commit_scope(&self) -> InteractionScope {
        InteractionScope::Global
    }
}

impl<AB> Air<AB> for MemoryLocalChip
where
    AB: SP1AirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &MemoryLocalCols<AB::Var> = (*local).borrow();

        for local in local.memory_local_entries.iter() {
            builder.assert_eq(
                local.is_real * local.is_real * local.is_real,
                local.is_real * local.is_real * local.is_real,
            );

            for scope in [InteractionScope::Global, InteractionScope::Local] {
                let mut values =
                    vec![local.initial_shard.into(), local.initial_clk.into(), local.addr.into()];
                values.extend(local.initial_value.map(Into::into));
                builder.receive(
                    AirInteraction::new(
                        values.clone(),
                        local.is_real.into(),
                        InteractionKind::Memory,
                    ),
                    scope,
                );

                let mut values =
                    vec![local.final_shard.into(), local.final_clk.into(), local.addr.into()];
                values.extend(local.final_value.map(Into::into));
                builder.send(
                    AirInteraction::new(
                        values.clone(),
                        local.is_real.into(),
                        InteractionKind::Memory,
                    ),
                    scope,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_matrix::dense::RowMajorMatrix;
    use sp1_core_executor::{programs::tests::simple_program, ExecutionRecord, Executor};
    use sp1_stark::{
        air::{InteractionScope, MachineAir},
        baby_bear_poseidon2::BabyBearPoseidon2,
        debug_interactions_with_all_chips, InteractionKind, SP1CoreOpts, StarkMachine,
    };

    use crate::{
        memory::MemoryLocalChip, riscv::RiscvAir,
        syscall::precompiles::sha256::extend_tests::sha_extend_program, utils::setup_logger,
    };

    #[test]
    fn test_local_memory_generate_trace() {
        let program = simple_program();
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run().unwrap();
        let shard = runtime.records[0].clone();

        let chip: MemoryLocalChip = MemoryLocalChip::new();

        let trace: RowMajorMatrix<BabyBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default());
        println!("{:?}", trace.values);

        for mem_event in shard.global_memory_finalize_events {
            println!("{:?}", mem_event);
        }
    }

    #[test]
    fn test_memory_lookup_interactions() {
        setup_logger();
        let program = sha_extend_program();
        let program_clone = program.clone();
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run().unwrap();
        let machine: StarkMachine<BabyBearPoseidon2, RiscvAir<BabyBear>> =
            RiscvAir::machine(BabyBearPoseidon2::new());
        let (pkey, _) = machine.setup(&program_clone);
        let opts = SP1CoreOpts::default();
        machine.generate_dependencies(&mut runtime.records, &opts, None);

        let shards = runtime.records;
        for shard in shards.clone() {
            debug_interactions_with_all_chips::<BabyBearPoseidon2, RiscvAir<BabyBear>>(
                &machine,
                &pkey,
                &[shard],
                vec![InteractionKind::Memory],
                InteractionScope::Local,
            );
        }
        debug_interactions_with_all_chips::<BabyBearPoseidon2, RiscvAir<BabyBear>>(
            &machine,
            &pkey,
            &shards,
            vec![InteractionKind::Memory],
            InteractionScope::Global,
        );
    }

    #[test]
    fn test_byte_lookup_interactions() {
        setup_logger();
        let program = sha_extend_program();
        let program_clone = program.clone();
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run().unwrap();
        let machine = RiscvAir::machine(BabyBearPoseidon2::new());
        let (pkey, _) = machine.setup(&program_clone);
        let opts = SP1CoreOpts::default();
        machine.generate_dependencies(&mut runtime.records, &opts, None);

        let shards = runtime.records;
        for shard in shards.clone() {
            debug_interactions_with_all_chips::<BabyBearPoseidon2, RiscvAir<BabyBear>>(
                &machine,
                &pkey,
                &[shard],
                vec![InteractionKind::Memory],
                InteractionScope::Local,
            );
        }
        debug_interactions_with_all_chips::<BabyBearPoseidon2, RiscvAir<BabyBear>>(
            &machine,
            &pkey,
            &shards,
            vec![InteractionKind::Byte],
            InteractionScope::Global,
        );
    }
}
