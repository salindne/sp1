use std::borrow::BorrowMut;

use p3_field::PrimeField32;
use p3_keccak_air::{generate_trace_rows, NUM_KECCAK_COLS, NUM_ROUNDS};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::{ParallelBridge, ParallelIterator, ParallelSlice};
use sp1_core_executor::{
    events::{ByteLookupEvent, KeccakPermuteEvent, PrecompileEvent, SyscallEvent},
    syscalls::SyscallCode,
    ExecutionRecord, Program,
};
use sp1_stark::air::MachineAir;

use crate::utils::zeroed_f_vec;

use super::{
    columns::{KeccakMemCols, NUM_KECCAK_MEM_COLS},
    KeccakPermuteChip, STATE_SIZE,
};
use sp1_core_executor::events::ByteRecord;

impl<F: PrimeField32> MachineAir<F> for KeccakPermuteChip {
    type Record = ExecutionRecord;
    type Program = Program;

    fn name(&self) -> String {
        "Keccak256".to_string()
    }

    // fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
    //     let chunk_size = std::cmp::max(input.keccak256_events.len() / num_cpus::get(), 1);

    //     let blu_batches = input
    //         .keccak256_events
    //         .par_chunks(chunk_size)
    //         .map(|events| {
    //             let mut blu: HashMap<u32, HashMap<ByteLookupEvent, usize>> = HashMap::new();
    //             events.iter().for_each(|event| {
    //                 let mut row = [F::zero(); NUM_KECCAK256_COLS];
    //                 let cols: &mut Keccak256Cols<F> = row.as_mut_slice().borrow_mut();
    //                 self.event_to_row(event, cols, &mut blu);
    //             });
    //             blu
    //         })
    //         .collect::<Vec<_>>();

    //     output.add_sharded_byte_lookup_events(blu_batches.iter().collect_vec());
    // }

    // fn generate_trace(
    //     &self,
    //     input: &ExecutionRecord,
    //     _output: &mut ExecutionRecord,
    // ) -> RowMajorMatrix<F> {
    //     // Generate the trace rows for each event.
    //     let mut rows = input
    //         .keccak256_events
    //         .iter()
    //         .map(|event| {
    //             let mut row = [F::zero(); NUM_KECCAK256_COLS];
    //             let cols: &mut Keccak256Cols<F> = row.as_mut_slice().borrow_mut();
    //             self.event_to_row(event, cols, &mut None);
    //             row
    //         })
    //         .collect::<Vec<_>>();

    //     // Pad the trace to a power of two depending on the proof shape in `input`.
    //     pad_rows_fixed(
    //         &mut rows,
    //         || [F::zero(); NUM_KECCAK256_COLS],
    //         input.fixed_log2_rows::<F, _>(self),
    //     );

    //     RowMajorMatrix::new(rows.into_iter().flatten().collect::<Vec<_>>(), NUM_KECCAK256_COLS)
    // }

    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            !shard.get_precompile_events(SyscallCode::KECCAK_PERMUTE).is_empty()
        }
    }
}

impl KeccakPermuteChip {
    pub fn populate_chunk<F: PrimeField32>(
        event: &KeccakPermuteEvent,
        chunk: &mut [F],
        new_byte_lookup_events: &mut Vec<ByteLookupEvent>,
    ) {
        let start_clk = event.clk;
        let shard = event.shard;

        let p3_keccak_trace = generate_trace_rows::<F>(vec![event.pre_state]);

        // Create all the rows for the permutation.
        for i in 0..NUM_ROUNDS {
            let p3_keccak_row = p3_keccak_trace.row(i);
            let row = &mut chunk[i * NUM_KECCAK_MEM_COLS..(i + 1) * NUM_KECCAK_MEM_COLS];
            // Copy p3_keccak_row into start of cols
            row[..NUM_KECCAK_COLS].copy_from_slice(p3_keccak_row.collect::<Vec<_>>().as_slice());
            let cols: &mut KeccakMemCols<F> = row.borrow_mut();

            cols.shard = F::from_canonical_u32(shard);
            cols.clk = F::from_canonical_u32(start_clk);
            cols.state_addr = F::from_canonical_u32(event.state_addr);
            cols.is_real = F::one();

            // If this is the first row, then populate read memory accesses
            if i == 0 {
                for (j, read_record) in event.state_read_records.iter().enumerate() {
                    cols.state_mem[j].populate_read(*read_record, new_byte_lookup_events);
                    new_byte_lookup_events
                        .add_u8_range_checks(shard, &read_record.value.to_le_bytes());
                }
                cols.do_memory_check = F::one();
                cols.receive_ecall = F::one();
            }

            // If this is the last row, then populate write memory accesses
            if i == NUM_ROUNDS - 1 {
                for (j, write_record) in event.state_write_records.iter().enumerate() {
                    cols.state_mem[j].populate_write(*write_record, new_byte_lookup_events);
                    new_byte_lookup_events
                        .add_u8_range_checks(shard, &write_record.value.to_le_bytes());
                }
                cols.do_memory_check = F::one();
            }
        }
    }
}
