// mlx_derive/src/lib.rs
//
// Proc macro crate for auto-deriving the `ModuleParams` trait.
//
// Architecture:
//   ModuleParams  — auto-derived, handles parameter traversal + updates + train mode
//   Module        — user implements, only requires forward()
//   Module: ModuleParams  (Module is a super-trait of ModuleParams)
//
// Field Attributes:
//   #[param]              — An `Array` that is a learnable parameter (weight, bias, etc.)
//   #[param(optional)]    — An `Option<Array>` learnable parameter
//   #[module]             — A sub-module field (any type implementing ModuleParams)
//   #[module(optional)]   — An `Option<T>` where T implements ModuleParams
//   #[state]              — A bool field toggled by train() (e.g., `training: bool`)
//
// What gets generated:
//   - parameters(&self) -> Vec<&Array>
//   - parameters_mut(&mut self) -> Vec<&mut Array>
//   - update_parameters(&mut self, new_params: &[Array])
//   - train(&mut self, training: bool)
//
// Usage:
//   #[derive(ModuleParams)]
//   struct Linear {
//       #[param]             weight: Array,
//       #[param(optional)]   bias: Option<Array>,
//       in_features: usize,
//   }
//
//   #[derive(ModuleParams)]
//   struct DecoderLayer {
//       #[module]            self_attn: MultiHeadAttention,
//       #[module(optional)]  cross_attn: Option<MultiHeadAttention>,
//   }

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput, Data, Fields, Ident, Field};

// ── Attribute classification ──────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
enum FieldKind {
    Param,
    ParamOptional,
    SubModule,
    SubModuleOptional,
    TrainState,
    None,
}

fn classify_field(field: &Field) -> FieldKind {
    for attr in &field.attrs {
        if attr.path().is_ident("param") {
            let is_optional = attr.parse_args::<Ident>()
                .map(|ident| ident == "optional")
                .unwrap_or(false);
            return if is_optional { FieldKind::ParamOptional } else { FieldKind::Param };
        }
        if attr.path().is_ident("module") {
            let is_optional = attr.parse_args::<Ident>()
                .map(|ident| ident == "optional")
                .unwrap_or(false);
            return if is_optional { FieldKind::SubModuleOptional } else { FieldKind::SubModule };
        }
        if attr.path().is_ident("state") {
            return FieldKind::TrainState;
        }
    }
    FieldKind::None
}

// ── The derive macro ──────────────────────────────────────────────────────

#[proc_macro_derive(ModuleParams, attributes(param, module, state))]
pub fn derive_module_params(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    let generics = &input.generics;
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let fields = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(named) => &named.named,
            _ => panic!("#[derive(ModuleParams)] only supports structs with named fields"),
        },
        _ => panic!("#[derive(ModuleParams)] only supports structs"),
    };

    let classified: Vec<(&Field, FieldKind)> = fields.iter()
        .map(|f| (f, classify_field(f)))
        .collect();

    // ── parameters(&self) ─────────────────────────────────────────────
    let params_items: Vec<_> = classified.iter().filter_map(|(f, kind)| {
        let ident = f.ident.as_ref().unwrap();
        match kind {
            FieldKind::Param => Some(quote! { params.push(&self.#ident); }),
            FieldKind::ParamOptional => Some(quote! {
                if let Some(ref p) = self.#ident { params.push(p); }
            }),
            FieldKind::SubModule => Some(quote! {
                params.extend(self.#ident.parameters());
            }),
            FieldKind::SubModuleOptional => Some(quote! {
                if let Some(ref m) = self.#ident { params.extend(m.parameters()); }
            }),
            _ => None,
        }
    }).collect();

    // ── parameters_mut(&mut self) ─────────────────────────────────────
    let params_mut_items: Vec<_> = classified.iter().filter_map(|(f, kind)| {
        let ident = f.ident.as_ref().unwrap();
        match kind {
            FieldKind::Param => Some(quote! { params.push(&mut self.#ident); }),
            FieldKind::ParamOptional => Some(quote! {
                if let Some(ref mut p) = self.#ident { params.push(p); }
            }),
            FieldKind::SubModule => Some(quote! {
                params.extend(self.#ident.parameters_mut());
            }),
            FieldKind::SubModuleOptional => Some(quote! {
                if let Some(ref mut m) = self.#ident { params.extend(m.parameters_mut()); }
            }),
            _ => None,
        }
    }).collect();

    // ── update_parameters(&mut self, new_params: &[Array]) ────────────
    let update_items: Vec<_> = classified.iter().filter_map(|(f, kind)| {
        let ident = f.ident.as_ref().unwrap();
        match kind {
            FieldKind::Param => Some(quote! {
                if offset < new_params.len() {
                    self.#ident = new_params[offset].clone();
                    offset += 1;
                }
            }),
            FieldKind::ParamOptional => Some(quote! {
                if self.#ident.is_some() && offset < new_params.len() {
                    self.#ident = Some(new_params[offset].clone());
                    offset += 1;
                }
            }),
            FieldKind::SubModule => Some(quote! {
                {
                    let n = self.#ident.parameters().len();
                    if offset + n <= new_params.len() {
                        self.#ident.update_parameters(&new_params[offset..offset + n]);
                        offset += n;
                    }
                }
            }),
            FieldKind::SubModuleOptional => Some(quote! {
                if let Some(ref mut m) = self.#ident {
                    let n = m.parameters().len();
                    if offset + n <= new_params.len() {
                        m.update_parameters(&new_params[offset..offset + n]);
                        offset += n;
                    }
                }
            }),
            _ => None,
        }
    }).collect();

    // ── train(&mut self, training: bool) ──────────────────────────────
    let train_items: Vec<_> = classified.iter().filter_map(|(f, kind)| {
        let ident = f.ident.as_ref().unwrap();
        match kind {
            FieldKind::TrainState => Some(quote! { self.#ident = training; }),
            FieldKind::SubModule => Some(quote! { self.#ident.train(training); }),
            FieldKind::SubModuleOptional => Some(quote! {
                if let Some(ref mut m) = self.#ident { m.train(training); }
            }),
            _ => None,
        }
    }).collect();

    // ── Assemble the impl ─────────────────────────────────────────────
    let expanded = quote! {
        impl #impl_generics crate::nn::ModuleParams for #name #ty_generics #where_clause {
            fn parameters(&self) -> Vec<&crate::Array> {
                let mut params = Vec::new();
                #(#params_items)*
                params
            }

            fn parameters_mut(&mut self) -> Vec<&mut crate::Array> {
                let mut params = Vec::new();
                #(#params_mut_items)*
                params
            }

            fn update_parameters(&mut self, new_params: &[crate::Array]) {
                #[allow(unused_mut)]
                let mut offset: usize = 0;
                #(#update_items)*
                let _ = offset;
            }

            fn train(&mut self, training: bool) {
                let _ = training; // suppress unused if no state/submodules
                #(#train_items)*
            }
        }
    };

    TokenStream::from(expanded)
}