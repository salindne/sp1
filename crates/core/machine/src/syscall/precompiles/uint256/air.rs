use crate::{
    memory::{value_as_limbs, MemoryReadCols, MemoryWriteCols},
    operations::field::field_op::FieldOpCols,
};

use crate::{
    air::MemoryAirBuilder,
    operations::{field::range::FieldLtCols, IsZeroOperation},
    utils::{
        limbs_from_access, limbs_from_prev_access, pad_rows_fixed, words_to_bytes_le,
        words_to_bytes_le_vec,
    },
};

use generic_array::GenericArray;
use num::{BigUint, One, Zero};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use sp1_core_executor::{
    events::{ByteRecord, FieldOperation, PrecompileEvent},
    syscalls::SyscallCode,
    ExecutionRecord, Program,
};
use sp1_curves::{
    params::{Limbs, NumLimbs, NumWords},
    uint256::U256Field,
};
use sp1_derive::AlignedBorrow;
use sp1_stark::{
    air::{BaseAirBuilder, InteractionScope, MachineAir, Polynomial, SP1AirBuilder},
    MachineRecord,
};
use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};
use typenum::Unsigned;

/// The number of columns in the Uint256MulCols.
const NUM_COLS: usize = size_of::<Uint256MulCols<u8>>();

#[derive(Default)]
pub struct Uint256MulChip;

impl Uint256MulChip {
    pub const fn new() -> Self {
        Self
    }
}

type WordsFieldElement = <U256Field as NumWords>::WordsFieldElement;
const WORDS_FIELD_ELEMENT: usize = WordsFieldElement::USIZE;

/// A set of columns for the Uint256Mul operation.
#[derive(Debug, Clone, AlignedBorrow)]
#[repr(C)]
pub struct Uint256MulCols<T> {
    /// The shard number of the syscall.
    pub shard: T,

    /// The clock cycle of the syscall.
    pub clk: T,

    /// The nonce of the operation.
    pub nonce: T,

    /// The pointer to the first input.
    pub x_ptr: T,

    /// The pointer to the second input, which contains the y value and the modulus.
    pub y_ptr: T,

    // Memory columns.
    // x_memory is written to with the result, which is why it is of type MemoryWriteCols.
    pub x_memory: GenericArray<MemoryWriteCols<T>, WordsFieldElement>,
    pub y_memory: GenericArray<MemoryReadCols<T>, WordsFieldElement>,
    pub modulus_memory: GenericArray<MemoryReadCols<T>, WordsFieldElement>,

    /// Columns for checking if modulus is zero. If it's zero, then use 2^256 as the effective
    /// modulus.
    pub modulus_is_zero: IsZeroOperation<T>,

    /// Column that is equal to is_real * (1 - modulus_is_zero.result).
    pub modulus_is_not_zero: T,

    // Output values. We compute (x * y) % modulus.
    pub output: FieldOpCols<T, U256Field>,

    pub output_range_check: FieldLtCols<T, U256Field>,

    pub is_real: T,
}

impl<F: PrimeField32> MachineAir<F> for Uint256MulChip {
    type Record = ExecutionRecord;
    type Program = Program;

    fn name(&self) -> String {
        "Uint256".to_string()
    }

    // fn generate_trace(
    //     &self,
    //     input: &Self::Record,
    //     output: &mut Self::Record,
    // ) -> RowMajorMatrix<F> {
    //     let mut rows = Vec::new();
    //     let mut new_byte_lookup_events = Vec::new();

    //     for (_, event) in input.get_precompile_events(SyscallCode::UINT256) {
    //         let event = if let PrecompileEvent::Uint256(event) = event {
    //             event
    //         } else {
    //             unreachable!();
    //         };

    //         let mut row = zeroed_f_vec(NUM_UINT256_COLS);
    //         let cols: &mut Uint256Cols<F> = row.as_mut_slice().borrow_mut();

    //         cols.is_real = F::one();
    //         cols.shard = F::from_canonical_u32(event.shard);
    //         cols.clk = F::from_canonical_u32(event.clk);
    //         cols.x_ptr = F::from_canonical_u32(event.x_ptr);
    //         cols.y_ptr = F::from_canonical_u32(event.y_ptr);
    //         cols.operation = F::from_canonical_u32(event.operation as u32);

    //         Self::populate_field_ops(
    //             &mut new_byte_lookup_events,
    //             event.shard,
    //             cols,
    //             event.x,
    //             event.y,
    //             event.operation,
    //         );

    //         // Populate the memory access columns.
    //         for i in 0..cols.y_access.len() {
    //             cols.y_access[i].populate(event.y_memory_records[i], &mut new_byte_lookup_events);
    //         }
    //         for i in 0..cols.x_access.len() {
    //             cols.x_access[i].populate(event.x_memory_records[i], &mut new_byte_lookup_events);
    //         }
    //         rows.push(row);
    //     }

    //     output.add_byte_lookup_events(new_byte_lookup_events);

    //     pad_rows_fixed(
    //         &mut rows,
    //         || {
    //             let mut row = zeroed_f_vec(NUM_UINT256_COLS);
    //             let cols: &mut Uint256Cols<F> = row.as_mut_slice().borrow_mut();
    //             Self::populate_field_ops(
    //                 &mut vec![],
    //                 0,
    //                 cols,
    //                 [0; WORD_SIZE],
    //                 [0; WORD_SIZE],
    //                 FieldOperation::Add,
    //             );
    //             row
    //         },
    //         input.fixed_log2_rows::<F, _>(self),
    //     );

    //     // Convert the trace to a row major matrix.
    //     let mut trace = RowMajorMatrix::new(rows.into_iter().flatten().collect::<Vec<_>>(), NUM_UINT256_COLS);

    //     // Write the nonces to the trace.
    //     for i in 0..trace.height() {
    //         let cols: &mut Uint256Cols<F> =
    //             trace.values[i * NUM_UINT256_COLS..(i + 1) * NUM_UINT256_COLS].borrow_mut();
    //         cols.nonce = F::from_canonical_usize(i);
    //     }

    //     trace
    // }

    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            !shard.get_precompile_events(SyscallCode::UINT256_MUL).is_empty()
        }
    }
}

impl<F> BaseAir<F> for Uint256MulChip {
    fn width(&self) -> usize {
        NUM_COLS
    }
}

impl<AB> Air<AB> for Uint256MulChip
where
    AB: SP1AirBuilder,
    Limbs<AB::Var, <U256Field as NumLimbs>::Limbs>: Copy,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &Uint256MulCols<AB::Var> = (*local).borrow();
        let next = main.row_slice(1);
        let next: &Uint256MulCols<AB::Var> = (*next).borrow();

        // Constrain the incrementing nonce.
        builder.when_first_row().assert_zero(local.nonce);
        builder.when_transition().assert_eq(local.nonce + AB::Expr::one(), next.nonce);

        // We are computing (x * y) % modulus. The value of x is stored in the "prev_value" of
        // the x_memory, since we write to it later.
        let x_limbs = limbs_from_prev_access(&local.x_memory);
        let y_limbs = limbs_from_access(&local.y_memory);
        let modulus_limbs = limbs_from_access(&local.modulus_memory);

        // If the modulus is zero, then we don't perform the modulus operation.
        // Evaluate the modulus_is_zero operation by summing each byte of the modulus. The sum will
        // not overflow because we are summing 32 bytes.
        let modulus_byte_sum =
            modulus_limbs.0.iter().fold(AB::Expr::zero(), |acc, &limb| acc + limb);
        IsZeroOperation::<AB::F>::eval(
            builder,
            modulus_byte_sum,
            local.modulus_is_zero,
            local.is_real.into(),
        );

        // If the modulus is zero, we'll actually use 2^256 as the modulus, so nothing happens.
        // Otherwise, we use the modulus passed in.
        let modulus_is_zero = local.modulus_is_zero.result;
        let mut coeff_2_256 = Vec::new();
        coeff_2_256.resize(32, AB::Expr::zero());
        coeff_2_256.push(AB::Expr::one());
        let modulus_polynomial: Polynomial<AB::Expr> = modulus_limbs.into();
        let p_modulus: Polynomial<AB::Expr> = modulus_polynomial
            * (AB::Expr::one() - modulus_is_zero.into())
            + Polynomial::from_coefficients(&coeff_2_256) * modulus_is_zero.into();

        // Evaluate the uint256 multiplication
        local.output.eval_with_modulus(
            builder,
            &x_limbs,
            &y_limbs,
            &p_modulus,
            FieldOperation::Mul,
            local.is_real,
        );

        // Verify the range of the output if the moduls is not zero.  Also, check the value of
        // modulus_is_not_zero.
        local.output_range_check.eval(
            builder,
            &local.output.result,
            &modulus_limbs,
            local.modulus_is_not_zero,
        );
        builder.assert_eq(
            local.modulus_is_not_zero,
            local.is_real * (AB::Expr::one() - modulus_is_zero.into()),
        );

        // Assert that the correct result is being written to x_memory.
        builder
            .when(local.is_real)
            .assert_all_eq(local.output.result, value_as_limbs(&local.x_memory));

        // Read and write x.
        builder.eval_memory_access_slice(
            local.shard,
            local.clk.into() + AB::Expr::one(),
            local.x_ptr,
            &local.x_memory,
            local.is_real,
        );

        // Evaluate the y_ptr memory access. We concatenate y and modulus into a single array since
        // we read it contiguously from the y_ptr memory location.
        builder.eval_memory_access_slice(
            local.shard,
            local.clk.into(),
            local.y_ptr,
            &[local.y_memory, local.modulus_memory].concat(),
            local.is_real,
        );

        // Receive the arguments.
        builder.receive_syscall(
            local.shard,
            local.clk,
            local.nonce,
            AB::F::from_canonical_u32(SyscallCode::UINT256_MUL.syscall_id()),
            local.x_ptr,
            local.y_ptr,
            local.is_real,
            InteractionScope::Local,
        );

        // Assert that is_real is a boolean.
        builder.assert_bool(local.is_real);
    }
}
